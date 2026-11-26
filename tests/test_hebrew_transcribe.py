"""Offline tests for hebrew-transcribe.

The speech and speaker models are replaced by small fakes, so nothing is
downloaded and no audio stack or GPU is needed:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
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
MODEL_ROOT = tempfile.mkdtemp(prefix="fake-models-")

SEGMENTS = [  # start, end, text, avg_logprob, no_speech_prob
    (0.5, 3.2, " שלום, רציתי לקבוע תור ", -0.21, 0.02),
    (3.4, 6.9, " בטח, איזה יום נוח לך? ", -0.71, 0.05),  # low confidence
    (7.0, 7.5, "   ", -0.3, 0.9),  # empty: dropped
    (8.0, 3725.25, " pipe | test ", None, None),
]
TURNS = [(0.0, 3.3, "SPEAKER_07"), (3.3, 7.0, "SPEAKER_02"), (7.0, 9999.0, "SPEAKER_07")]

# What the fake language detector "hears", by the first bytes of a file.
DETECTION = {
    b"ru": ("ru", 0.97, [("ru", 0.97), ("uk", 0.02), ("he", 0.01)]),
    b"en": ("en", 0.95, [("en", 0.95), ("he", 0.03), ("ru", 0.02)]),
    b"uk": ("uk", 0.60, [("uk", 0.60), ("ru", 0.35), ("he", 0.05)]),
}
HEBREW = ("he", 0.99, [("he", 0.99), ("ar", 0.005), ("ru", 0.005)])


class FakeAudio(list):
    """Decoded audio that remembers which file it came from."""
    source = b""


class FakeWhisperModel:
    last_kwargs: dict = {}

    def __init__(self, model_dir, device="cpu", compute_type="int8", cpu_threads=4):
        assert os.path.isdir(model_dir), model_dir
        self.name = Path(model_dir).name.replace("__", "/")
        STATE["loaded"].append(self.name)

    def detect_language(self, audio, vad_filter=False, vad_parameters=None,
                        language_detection_segments=1, language_detection_threshold=0.5):
        STATE["detections"].append(self.name)
        if STATE.get("no_speech"):
            raise ValueError("need at least one array to concatenate")
        return DETECTION.get(bytes(audio.source[:2]), HEBREW)

    def transcribe(self, audio, **kwargs):
        FakeWhisperModel.last_kwargs = dict(kwargs, model=self.name)
        if STATE.get("interrupt"):
            def interrupted():
                yield NS(start=0.0, end=1.0, text="x", avg_logprob=-0.1, no_speech_prob=0.0)
                raise KeyboardInterrupt
            return interrupted(), NS(language="he")
        segments = (NS(start=s, end=e, text=t, avg_logprob=lp, no_speech_prob=ns)
                    for s, e, t, lp, ns in SEGMENTS)
        return segments, NS(language=kwargs.get("language") or "he", duration=30.0)


def fake_decode_audio(path, sampling_rate=16000, split_stereo=False):
    audio = FakeAudio([0.0] * (sampling_rate * 30))  # 30 seconds
    with open(path, "rb") as f:  # read-only, like the real one
        audio.source = f.read()
    return audio


class RepositoryNotFoundError(Exception):
    pass


def fake_snapshot_download(repo_id, local_files_only=False, **kwargs):
    mode = STATE["hf"]
    if local_files_only and mode != "cached":
        raise FileNotFoundError("not in the cache")
    if not local_files_only:
        STATE["downloads"].append(repo_id)
        if mode == "notfound":
            raise RepositoryNotFoundError(repo_id)
        if mode == "network":
            raise ConnectionError("403 Forbidden")
    folder = os.path.join(MODEL_ROOT, repo_id.replace("/", "__"))
    os.makedirs(folder, exist_ok=True)
    return folder


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
                            HF_TOKEN_PATH=os.path.join(MODEL_ROOT, "token"))
    _module("ctranslate2", get_cuda_device_count=lambda: 0)
    _module("torch", from_numpy=lambda a: NS(unsqueeze=lambda dim: a), device=lambda name: name)
    pyannote = _module("pyannote")
    pyannote.audio = _module("pyannote.audio", Pipeline=FakePipeline)


install_fakes()

import hebrew_transcribe as ht  # noqa: E402
from hebrew_transcribe import cli, core, i18n, output  # noqa: E402

HE_TURBO = ht.LANGUAGE_MODELS["he"]["turbo"]
RU_TURBO = ht.LANGUAGE_MODELS["ru"]["turbo"]
RU_FULL = ht.LANGUAGE_MODELS["ru"]["full"]
DETECTOR = ht.DETECTION_MODEL


# ----------------------------------------------------------------------------
class Base(unittest.TestCase):
    def setUp(self):
        STATE.clear()
        STATE.update(pyannote="v4", hf="cached", token=None, interrupt=False, no_speech=False,
                     pipeline_calls=[], loaded=[], detections=[], downloads=[])
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "TRANSCRIBE_UI_LANGUAGE"):
            os.environ.pop(var, None)
        ht.set_ui_language("en")
        self.addCleanup(ht.set_ui_language, "en")

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.rec = self.tmp / "rec"
        (self.rec / "sub").mkdir(parents=True)
        files = {"call1.m4a": b"he-a", "מקליט 2.mp3": b"he-b", "sub/call1.m4a": b"he-c",
                 ".hidden.m4a": b"h", "notes.txt": b"n", "russian.m4a": b"ru-1",
                 "english.m4a": b"en-1", "ukrainian.m4a": b"uk-1"}
        for relative, data in files.items():
            (self.rec / relative).write_bytes(data)
        self.out = self.tmp / "out"

    def run_cli(self, *args) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([str(a) for a in args])
        return code, buffer.getvalue()


class HelperTests(Base):
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

    def test_parse_language(self):
        self.assertEqual(core.parse_language("he"), ("he", None))
        self.assertEqual(core.parse_language("RU"), ("ru", None))
        self.assertEqual(core.parse_language("auto"), (None, None))
        self.assertEqual(core.parse_language(None), (None, None))
        self.assertEqual(core.parse_language("he,ru"), (None, ("he", "ru")))
        self.assertEqual(core.parse_language(["russian", "iw"]), (None, ("ru", "he")))
        self.assertEqual(core.parse_language("русский"), ("ru", None))
        with self.assertRaises(ValueError):
            core.parse_language("klingon")

    def test_models_per_language(self):
        self.assertEqual(ht.model_for("he"), HE_TURBO)
        self.assertEqual(ht.model_for("ru"), RU_TURBO)
        self.assertEqual(ht.model_for("ru", "full"), RU_FULL)
        self.assertEqual(ht.model_for("fr"), DETECTOR)
        self.assertEqual(ht.MODELS, ht.LANGUAGE_MODELS["he"])  # the 1.x name still works
        self.assertEqual(ht.required_models("turbo", "he"), [HE_TURBO])
        self.assertEqual(ht.required_models("turbo", "auto"), [DETECTOR, HE_TURBO, RU_TURBO])
        self.assertEqual(ht.required_models("turbo", "ru,he"), [DETECTOR, RU_TURBO, HE_TURBO])
        self.assertEqual(ht.required_models("some/model", "auto"), ["some/model"])

    def test_translations_are_complete(self):
        english, russian = i18n.MESSAGES["en"], i18n.MESSAGES["ru"]
        self.assertEqual(set(english), set(russian))
        fields = re.compile(r"\{(\w+)\}")
        for key in english:
            self.assertEqual(set(fields.findall(english[key])), set(fields.findall(russian[key])), key)


class TranscriberTests(Base):
    def test_hebrew_by_default(self):
        result = ht.Transcriber().transcribe(self.rec / "call1.m4a")
        self.assertEqual(STATE["loaded"], [HE_TURBO])
        self.assertEqual(result.model, HE_TURBO)
        self.assertEqual(len(result.segments), 3)
        self.assertEqual([s.low_confidence for s in result.segments], [False, True, False])
        self.assertEqual(result.flagged, 1)
        self.assertIsNone(result.diarization)
        self.assertEqual(result.sha256, hashlib.sha256(b"he-a").hexdigest())
        self.assertEqual(result.duration, 30.0)
        self.assertEqual((result.language, result.detected_language), ("he", "he"))
        self.assertIsNone(result.language_probability)
        self.assertTrue(result.text.startswith("שלום"))
        kwargs = FakeWhisperModel.last_kwargs
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertIs(kwargs["vad_filter"], True)
        self.assertEqual(kwargs["language"], "he")
        self.assertIsNone(kwargs["initial_prompt"])
        self.assertEqual(STATE["detections"], [])

        txt = result.save(self.out)
        self.assertEqual(txt.name, "call1.txt")
        content = txt.read_text(encoding="utf-8")
        self.assertIn(ht.UNSURE, content)
        self.assertIn("not separated", content)
        self.assertNotIn("Speaker 1:", content)
        self.assertIn("01:02:05,250", (self.out / "call1.srt").read_text(encoding="utf-8"))
        self.assertEqual(ht.Transcript.load(self.out / "call1.json"), result)
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_russian_uses_the_russian_model(self):
        result = ht.Transcriber(language="ru").transcribe(self.rec / "russian.m4a")
        self.assertEqual(STATE["loaded"], [RU_TURBO])
        self.assertEqual(FakeWhisperModel.last_kwargs["language"], "ru")
        self.assertEqual((result.model, result.language, result.detected_language), (RU_TURBO, "ru", "ru"))
        ht.Transcriber("full", language="ru")
        self.assertEqual(STATE["loaded"][-1], RU_FULL)

    def test_auto_picks_the_model_for_each_file(self):
        transcriber = ht.Transcriber(language="auto")
        self.assertEqual(STATE["loaded"], [DETECTOR])  # only the detector up front
        expected = {"call1.m4a": ("he", HE_TURBO), "russian.m4a": ("ru", RU_TURBO),
                    "english.m4a": ("en", DETECTOR), "מקליט 2.mp3": ("he", HE_TURBO)}
        for name, (language, model) in expected.items():
            result = transcriber.transcribe(self.rec / name)
            self.assertEqual((result.detected_language, result.model), (language, model), name)
            self.assertEqual(result.language, "auto")
            self.assertGreater(result.language_probability, 0.9)
            self.assertEqual(FakeWhisperModel.last_kwargs["language"], language)
        self.assertEqual(STATE["loaded"], [DETECTOR, HE_TURBO, RU_TURBO])  # each loaded once
        self.assertEqual(set(STATE["detections"]), {DETECTOR})
        txt = output.render_txt(result)
        self.assertIn("auto → he (99%)", txt)

    def test_candidate_languages_limit_detection(self):
        transcriber = ht.Transcriber(language="he,ru")
        with self.assertLogs("hebrew_transcribe", level="INFO") as logs:
            result = transcriber.transcribe(self.rec / "ukrainian.m4a")
        self.assertEqual((result.detected_language, result.model), ("ru", RU_TURBO))
        self.assertEqual(result.language, "he,ru")
        self.assertAlmostEqual(result.language_probability, 0.35)
        self.assertTrue(any("uncertain" in line for line in logs.output))

    def test_no_speech_falls_back_without_crashing(self):
        STATE["no_speech"] = True
        with self.assertLogs("hebrew_transcribe", level="WARNING"):
            result = ht.Transcriber(language="ru,he").transcribe(self.rec / "russian.m4a")
        self.assertEqual((result.detected_language, result.language_probability), ("ru", 0.0))

    def test_custom_model_detects_for_itself(self):
        result = ht.Transcriber("some/model", language="auto").transcribe(self.rec / "russian.m4a")
        self.assertEqual(STATE["loaded"], ["some/model"])
        self.assertEqual(STATE["detections"], ["some/model"])
        self.assertEqual((result.model, result.detected_language), ("some/model", "ru"))

    def test_prompts(self):
        transcriber = ht.Transcriber(prompt="רונית")
        transcriber.transcribe(self.rec / "call1.m4a")
        self.assertEqual(FakeWhisperModel.last_kwargs["initial_prompt"], "רונית")
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
        self.assertTrue(ht.resolve_model("turbo").endswith(HE_TURBO.replace("/", "__")))
        self.assertEqual(ht.resolve_model(str(self.tmp)), str(self.tmp))  # local folder
        STATE["hf"] = "download"
        self.assertTrue(ht.resolve_model(RU_TURBO).endswith(RU_TURBO.replace("/", "__")))
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
        self.assertEqual(len(list(self.out.glob("*.txt"))), 6)
        self.assertIn("rec__call1.txt", {p.name for p in self.out.glob("*.txt")})
        index = (self.out / "INDEX.md").read_text(encoding="utf-8")
        self.assertEqual(index.count(".txt` |"), 6)

        code, text = self.run_cli(self.rec, "--recursive", "--out", self.out)
        self.assertEqual(code, 0)
        self.assertEqual(text.count("already transcribed"), 6)

        code, text = self.run_cli(self.rec / "call1.m4a", "--out", self.out, "--force", "-q")
        self.assertEqual(code, 0)
        self.assertIn("done: call1.txt", text)

    def test_auto_language_for_a_folder(self):
        code, text = self.run_cli(self.rec, "--out", self.out, "--language", "auto", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("Loading the language-detection model", text)
        data = json.loads((self.out / "russian.json").read_text(encoding="utf-8"))
        self.assertEqual((data["detected_language"], data["model"]), ("ru", RU_TURBO))
        self.assertIn("ru (97%)", (self.out / "INDEX.md").read_text(encoding="utf-8"))

    def test_russian_messages_and_files(self):
        code, text = self.run_cli(self.rec / "russian.m4a", "--out", self.out, "--language", "ru",
                                  "--diarize", "--ui-language", "ru", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("готово", text)
        self.assertIn("Завершено.", text)
        txt = (self.out / "russian.txt").read_text(encoding="utf-8")
        self.assertIn("Расшифровка — russian.m4a", txt)
        self.assertIn("Спикер 1:", txt)
        self.assertIn("Спикер 2:", (self.out / "russian.srt").read_text(encoding="utf-8"))
        data = json.loads((self.out / "russian.json").read_text(encoding="utf-8"))
        self.assertEqual(data["segments"][0]["speaker"], "Speaker 1")  # data stays the same
        index = (self.out / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("# Расшифровки", index)
        self.assertIn("| да |", index)

    def test_russian_from_the_environment(self):
        os.environ["TRANSCRIBE_UI_LANGUAGE"] = "ru"
        code, text = self.run_cli(self.tmp / "missing.m4a", "--out", self.out)
        self.assertEqual(code, 1)
        self.assertIn("Нечего расшифровывать", text)

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
        STATE["hf"] = "download"
        code, text = self.run_cli("--download-only", "--language", "auto")
        self.assertEqual(code, 0, text)
        self.assertEqual(STATE["downloads"], [DETECTOR, HE_TURBO, RU_TURBO])
        self.assertEqual(text.count("Model ready"), 3)
        STATE["hf"] = "notfound"
        code, text = self.run_cli("--download-only", "--model", "nobody/nothing")
        self.assertEqual(code, 1)

    def test_help_version_and_bad_language(self):
        for args in (["--help"], ["--version"], ["--ui-language", "ru", "--help"]):
            out = io.StringIO()
            with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(out):
                cli.parse_args(args)
            self.assertEqual(caught.exception.code, 0)
        self.assertIn("параметры", out.getvalue())
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            cli.parse_args(["x.m4a", "--language", "klingon"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
