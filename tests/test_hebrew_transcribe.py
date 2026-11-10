"""Offline tests for hebrew-transcribe.

The speech and speaker models are replaced by small fakes, so nothing is
downloaded and no audio stack or GPU is needed:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

# ----------------------------------------------------------------------------
# Fakes for faster-whisper, huggingface_hub, ctranslate2, torch, pyannote.audio
# ----------------------------------------------------------------------------
STATE: dict = {}
MODEL_DIR = tempfile.mkdtemp(prefix="fake-model-")

SEGMENTS = [  # start, end, text, avg_logprob, no_speech_prob
    (0.5, 3.2, " שלום, רציתי לקבוע תור ", -0.21, 0.02),
    (3.4, 6.9, " בטח, איזה יום נוח לך? ", -0.71, 0.05),  # low confidence
    (7.0, 7.5, "   ", -0.3, 0.9),  # empty: dropped
    (8.0, 3725.25, " pipe | test ", None, None),
]
TURNS = [(0.0, 3.3, "SPEAKER_07"), (3.3, 7.0, "SPEAKER_02"), (7.0, 9999.0, "SPEAKER_07")]


class FakeWhisperModel:
    last_kwargs: dict = {}

    def __init__(self, model_dir, device="cpu", compute_type="int8", cpu_threads=4):
        assert os.path.isdir(model_dir), model_dir

    def transcribe(self, audio, **kwargs):
        FakeWhisperModel.last_kwargs = kwargs
        if STATE.get("interrupt"):
            def interrupted():
                yield NS(start=0.0, end=1.0, text="x", avg_logprob=-0.1, no_speech_prob=0.0)
                raise KeyboardInterrupt
            return interrupted(), NS(language="he")
        segments = (NS(start=s, end=e, text=t, avg_logprob=lp, no_speech_prob=ns)
                    for s, e, t, lp, ns in SEGMENTS)
        return segments, NS(language="he", language_probability=0.99, duration=30.0)


def fake_decode_audio(path, sampling_rate=16000, split_stereo=False):
    with open(path, "rb") as f:  # read-only, like the real one
        f.read()
    return [0.0] * (sampling_rate * 30)  # 30 seconds


class RepositoryNotFoundError(Exception):
    pass


def fake_snapshot_download(repo_id, local_files_only=False, **kwargs):
    mode = STATE["hf"]
    if local_files_only and mode != "cached":
        raise FileNotFoundError("not in the cache")
    if not local_files_only and mode == "notfound":
        raise RepositoryNotFoundError(repo_id)
    if not local_files_only and mode == "network":
        raise ConnectionError("403 Forbidden")
    return MODEL_DIR


class FakeAnnotation:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        for i, (start, end, label) in enumerate(self.tracks):
            yield NS(start=start, end=end), f"T{i}", label


class FakePipelineInstance:
    def __init__(self, version):
        self.version = version

    def to(self, device):
        pass

    def __call__(self, data, **kwargs):
        assert set(data) == {"waveform", "sample_rate"}
        STATE["pipeline_calls"].append(kwargs)
        if self.version == "v4":
            return NS(speaker_diarization=FakeAnnotation([(0, 1, "X")]),
                      exclusive_speaker_diarization=FakeAnnotation(TURNS))
        return FakeAnnotation(TURNS)


class FakePipeline:
    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        STATE["from_pretrained"] = (checkpoint, sorted(kwargs))
        mode = STATE["pyannote"]
        if mode == "v3" and "token" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'token'")
        if mode == "deny":
            return None
        return FakePipelineInstance(mode)


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def install_fakes():
    fw = _module("faster_whisper", WhisperModel=FakeWhisperModel)
    fw.audio = _module("faster_whisper.audio", decode_audio=fake_decode_audio)
    hub = _module("huggingface_hub", snapshot_download=fake_snapshot_download,
                  get_token=lambda: STATE["token"], RepositoryNotFoundError=RepositoryNotFoundError)
    hub.constants = _module("huggingface_hub.constants",
                            HF_TOKEN_PATH=os.path.join(MODEL_DIR, "token"))
    _module("ctranslate2", get_cuda_device_count=lambda: 0)
    _module("torch", from_numpy=lambda a: NS(unsqueeze=lambda dim: a), device=lambda name: name)
    pyannote = _module("pyannote")
    pyannote.audio = _module("pyannote.audio", Pipeline=FakePipeline)


install_fakes()

import hebrew_transcribe as ht  # noqa: E402
from hebrew_transcribe import cli, core, output  # noqa: E402


# ----------------------------------------------------------------------------
class Base(unittest.TestCase):
    def setUp(self):
        STATE.clear()
        STATE.update(pyannote="v4", hf="cached", token=None, interrupt=False, pipeline_calls=[])
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            os.environ.pop(var, None)

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = self.tmp / "rec"
        (self.rec / "sub").mkdir(parents=True)
        files = {"call1.m4a": b"a", "מקליט 2.mp3": b"b", "sub/call1.m4a": b"c",
                 ".hidden.m4a": b"h", "notes.txt": b"n"}
        for relative, data in files.items():
            (self.rec / relative).write_bytes(data)
        self.out = self.tmp / "out"

    def run_cli(self, *args) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([str(a) for a in args])
        return code, buffer.getvalue()


class HelperTests(unittest.TestCase):
    def test_clock_and_srt_time(self):
        self.assertEqual(core.clock(12.3), "00:12")
        self.assertEqual(core.clock(3725), "1:02:05")
        self.assertEqual(core.srt_time(3.5), "00:00:03,500")
        self.assertEqual(core.srt_time(1.9996), "00:00:02,000")
        self.assertEqual(core.srt_time(3725.25), "01:02:05,250")

    def test_speaker_for(self):
        self.assertIsNone(core.speaker_for(0, 1, []))
        self.assertEqual(core.speaker_for(2, 5, [(0, 3, "A"), (3, 10, "B")]), "B")

    def test_output_names_are_unique(self):
        names = output.output_names([Path("a/x.m4a"), Path("b/x.m4a"), Path("a/y.mp3"), Path("a/x.mp3")])
        self.assertEqual(len(set(names.values())), 4)
        self.assertEqual(names[Path("a/y.mp3")], "y")

    def test_privacy_defaults_on_import(self):
        self.assertIn("PYANNOTE_METRICS_ENABLED", os.environ)
        self.assertIn("HF_HUB_DISABLE_TELEMETRY", os.environ)

    def test_public_api(self):
        for name in ht.__all__:
            self.assertTrue(hasattr(ht, name), name)


class TranscriberTests(Base):
    def test_transcribe_and_save(self):
        result = ht.Transcriber().transcribe(self.rec / "call1.m4a")
        self.assertEqual(len(result.segments), 3)
        self.assertEqual([s.low_confidence for s in result.segments], [False, True, False])
        self.assertEqual(result.flagged, 1)
        self.assertIsNone(result.diarization)
        self.assertEqual(result.sha256, hashlib.sha256(b"a").hexdigest())
        self.assertEqual(result.duration, 30.0)
        self.assertEqual(result.language, "he")
        self.assertTrue(result.text.startswith("שלום"))
        kwargs = FakeWhisperModel.last_kwargs
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertIs(kwargs["vad_filter"], True)
        self.assertEqual(kwargs["language"], "he")
        self.assertIsNone(kwargs["initial_prompt"])

        txt = result.save(self.out)
        self.assertEqual(txt.name, "call1.txt")
        content = txt.read_text(encoding="utf-8")
        self.assertIn(ht.UNSURE, content)
        self.assertIn("not separated", content)
        self.assertNotIn("Speaker 1:", content)
        self.assertIn("01:02:05,250", (self.out / "call1.srt").read_text(encoding="utf-8"))
        self.assertEqual(ht.Transcript.load(self.out / "call1.json"), result)
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_language_auto_and_prompts(self):
        transcriber = ht.Transcriber(language="auto", prompt="רונית")
        result = transcriber.transcribe(self.rec / "call1.m4a")
        self.assertIsNone(FakeWhisperModel.last_kwargs["language"])
        self.assertEqual(FakeWhisperModel.last_kwargs["initial_prompt"], "רונית")
        self.assertEqual(result.language, "auto")
        transcriber.transcribe(self.rec / "call1.m4a", prompt="יוסי")
        self.assertEqual(FakeWhisperModel.last_kwargs["initial_prompt"], "יוסי")

    def test_on_segment_callback(self):
        seen = []
        ht.Transcriber().transcribe(self.rec / "call1.m4a", on_segment=seen.append)
        self.assertEqual(len(seen), 3)

    def test_speaker_labels_pyannote4(self):
        result = ht.Transcriber(diarize=True, hf_token="hf_dummy").transcribe(
            self.rec / "call1.m4a", speakers=2)
        self.assertEqual([s.speaker for s in result.segments], ["Speaker 1", "Speaker 2", "Speaker 1"])
        self.assertEqual(STATE["pipeline_calls"][-1], {"num_speakers": 2})
        self.assertEqual(STATE["from_pretrained"][1], ["token"])
        self.assertEqual(result.diarization, core.default_diarization_model())
        self.assertIn("Speaker 2:", result.to_txt())

    def test_speaker_labels_pyannote3_fallback(self):
        STATE["pyannote"] = "v3"
        diarizer = ht.Diarizer("hf_dummy", "pyannote/speaker-diarization-3.1")
        self.assertEqual(STATE["from_pretrained"][1], ["use_auth_token"])
        result = ht.Transcriber(diarizer=diarizer).transcribe(self.rec / "call1.m4a")
        self.assertEqual([s.speaker for s in result.segments], ["Speaker 1", "Speaker 2", "Speaker 1"])

    def test_diarization_unavailable(self):
        STATE["pyannote"] = "deny"
        with self.assertRaises(ht.DiarizationUnavailable):
            ht.Diarizer()
        STATE["pyannote"] = "v4"
        with self.assertRaises(ht.DiarizationUnavailable):  # would upload the audio
            ht.Diarizer(model="pyannote/speaker-diarization-precision-2")
        with mock.patch.dict(sys.modules, {"pyannote.audio": None}):  # not installed
            with self.assertRaises(ht.DiarizationUnavailable):
                ht.Diarizer()

    def test_model_resolution_and_errors(self):
        self.assertEqual(ht.resolve_model("turbo"), MODEL_DIR)  # cached: no download
        self.assertEqual(ht.resolve_model(str(self.tmp)), str(self.tmp))  # local folder
        STATE["hf"] = "download"
        self.assertEqual(ht.resolve_model("full"), MODEL_DIR)
        STATE["hf"] = "notfound"
        with self.assertRaises(ht.ModelDownloadError):
            ht.resolve_model("nobody/nothing")
        STATE["hf"] = "network"
        with mock.patch.object(core.time, "sleep"):
            with self.assertRaises(ht.ModelDownloadError) as caught:
                ht.resolve_model("turbo", attempts=2)
        self.assertIn("403", str(caught.exception))

    def test_source_file_is_never_modified(self):
        source = self.rec / "call1.m4a"
        before = (source.read_bytes(), source.stat().st_mtime_ns)
        ht.Transcriber().transcribe(source).save(self.rec)  # even when saving next to it
        self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before)


class CliTests(Base):
    def test_folder_resume_force_and_index(self):
        code, text = self.run_cli(self.rec, "--recursive", "--out", self.out, "-q")
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(p.name for p in self.out.glob("*.txt")),
                         sorted(["rec__call1.txt", "sub__call1.txt", "מקליט 2.txt"]))
        index = (self.out / "INDEX.md").read_text(encoding="utf-8")
        self.assertEqual(index.count(".txt` |"), 3)

        code, text = self.run_cli(self.rec, "--recursive", "--out", self.out)
        self.assertEqual(code, 0)
        self.assertEqual(text.count("already transcribed"), 3)

        code, text = self.run_cli(self.rec / "call1.m4a", "--out", self.out, "--force", "-q")
        self.assertEqual(code, 0)
        self.assertIn("done: call1.txt", text)

    def test_speaker_labels(self):
        code, text = self.run_cli(self.rec / "call1.m4a", "--out", self.out, "--diarize",
                                  "--speakers", "2", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("Speaker 1:", (self.out / "call1.txt").read_text(encoding="utf-8"))

    def test_speaker_labels_unavailable_continues(self):
        STATE["pyannote"] = "deny"
        code, text = self.run_cli(self.rec / "call1.m4a", "--out", self.out, "--diarize", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("Continuing without speaker labels", text)
        self.assertIn("not separated", (self.out / "call1.txt").read_text(encoding="utf-8"))

    def test_never_overwrites_an_input(self):
        keep = self.tmp / "x.txt"
        keep.write_text("keep me", encoding="utf-8")
        code, text = self.run_cli(keep, "--out", self.tmp, "--force")
        self.assertEqual(code, 1)
        self.assertEqual(keep.read_text(encoding="utf-8"), "keep me")

    def test_nothing_to_transcribe(self):
        code, text = self.run_cli(self.tmp / "missing.m4a", "--out", self.out)
        self.assertEqual(code, 1)
        self.assertIn("Nothing to transcribe", text)

    def test_interrupt_leaves_no_partial_files(self):
        STATE["interrupt"] = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli(self.rec / "call1.m4a", "--out", self.out, "-q")
        self.assertEqual(list(self.out.glob("call1.*")), [])

    def test_download_only(self):
        code, text = self.run_cli("--download-only")
        self.assertEqual(code, 0)
        self.assertIn("Model ready", text)
        STATE["hf"] = "notfound"
        code, text = self.run_cli("--download-only", "--model", "nobody/nothing")
        self.assertEqual(code, 1)

    def test_help_and_version(self):
        for flag in ("--help", "--version"):
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(io.StringIO()):
                cli.parse_args([flag])
            self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
