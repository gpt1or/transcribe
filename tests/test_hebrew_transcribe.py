"""Offline tests for hebrew-transcribe.

The speech and speaker models, the voice detector and the media decoder are
replaced by small fakes, so nothing is downloaded and no audio stack or GPU is
needed:

    python -m unittest discover -s tests -v

A fake recording is either a few bytes (b"ru-1": 30 seconds of Russian; the
first two bytes give the language, Hebrew by default) or a scripted
conversation written by script(): who speaks which language when.
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
# Fakes for faster-whisper, PyAV, huggingface_hub, ctranslate2, torch, pyannote.audio
# ----------------------------------------------------------------------------
STATE: dict = {}
MODEL_ROOT = tempfile.mkdtemp(prefix="fake-models-")
SR = 16_000
MEDIA = Path(__file__).resolve().parent / "media"

SEGMENTS = [  # what the fake model "says" for a plain recording: start, end, text, avg_logprob, no_speech_prob
    (0.5, 3.2, " שלום, רציתי לקבוע תור ", -0.21, 0.02),
    (3.4, 6.9, " בטח, איזה יום נוח לך? ", -0.71, 0.05),  # low confidence
    (7.0, 7.5, "   ", -0.3, 0.9),  # empty: dropped
    (8.0, 3725.25, " pipe | test ", None, None),
]
TURNS = [(0.0, 3.3, "SPEAKER_07"), (3.3, 7.0, "SPEAKER_02"), (7.0, 9999.0, "SPEAKER_07")]

# What the fake language detector "hears" in speech of each language.
DETECTION = {
    "he": ("he", 0.99, [("he", 0.99), ("ar", 0.005), ("ru", 0.005)]),
    "ru": ("ru", 0.97, [("ru", 0.97), ("uk", 0.02), ("he", 0.01)]),
    "en": ("en", 0.95, [("en", 0.95), ("he", 0.03), ("ru", 0.02)]),
    "uk": ("uk", 0.60, [("uk", 0.60), ("ru", 0.35), ("he", 0.05)]),
    "noise": ("en", 0.62, [("en", 0.62), ("ja", 0.2), ("he", 0.1), ("ru", 0.05)]),  # a laugh
}
SILENCE = ("en", 0.3, [("en", 0.3), ("nn", 0.2), ("he", 0.1), ("ru", 0.05)])


class FakeAudio:
    """Decoded audio: a length and a script of (start, end, language) in
    seconds. A slice remembers where it lies in the file."""

    def __init__(self, samples, speech, plain, offset=0):
        self.samples, self.speech, self.plain, self.offset = samples, speech, plain, offset

    def __len__(self):
        return self.samples

    def __getitem__(self, item):
        start, stop, step = item.indices(self.samples)
        assert step == 1
        return FakeAudio(max(0, stop - start), self.speech, self.plain, self.offset + start)

    def heard(self):
        """The speech in this piece, in seconds from the start of the piece."""
        a, b = self.offset / SR, (self.offset + self.samples) / SR
        return [(max(s, a) - a, min(e, b) - a, language)
                for s, e, language in self.speech if min(e, b) > max(s, a)]


def script(path: Path, speech, length=None) -> Path:
    """Write a fake recording of a conversation: [(start, end, language)]."""
    if length is None:
        length = max((end for _, end, _ in speech), default=9.0) + 1
    path.write_text(json.dumps({"length": length, "speech": speech}), encoding="utf-8")
    return path


def fake_decode_audio(path, sampling_rate=16000, split_stereo=False):
    with open(path, "rb") as f:  # read-only, like the real one
        data = f.read()
    if data.startswith(b"{"):
        spec = json.loads(data)
        return FakeAudio(int(spec["length"] * sampling_rate),
                         [tuple(item) for item in spec["speech"]], plain=False)
    language = data[:2].decode("ascii", "replace")
    language = language if language in DETECTION else "he"
    speech = [(0.5, 9.5, language), (10.5, 19.5, language), (20.5, 29.5, language)]
    return FakeAudio(30 * sampling_rate, speech, plain=True)


class FakeVadOptions:
    def __init__(self, **options):
        STATE["vad_options"] = options


def fake_speech_timestamps(audio, vad_options=None, sampling_rate=16000, **kwargs):
    """Speech with less than half a second between it is one chunk, as with the real one."""
    STATE["vad"] += 1
    if STATE.get("vad_broken"):
        raise RuntimeError("the voice detector cannot run")
    chunks: list[list[float]] = []
    for start, end, _ in audio.heard():
        if chunks and start - chunks[-1][1] < 0.5:
            chunks[-1][1] = end
        else:
            chunks.append([start, end])
    return [{"start": int(s * sampling_rate), "end": int(e * sampling_rate)} for s, e in chunks]


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
        spoken: dict[str, float] = {}
        for start, end, language in audio.heard():
            spoken[language] = spoken.get(language, 0.0) + end - start
        return DETECTION[max(spoken, key=spoken.get)] if spoken else SILENCE

    def transcribe(self, audio, **kwargs):
        language = kwargs.get("language")
        STATE["transcribed"].append(
            (self.name, language, round(audio.offset / SR, 3), round(len(audio) / SR, 3)))
        FakeWhisperModel.last_kwargs = dict(kwargs, model=self.name)
        if STATE.get("interrupt"):
            def interrupted():
                yield NS(start=0.0, end=1.0, text="x", avg_logprob=-0.1, no_speech_prob=0.0)
                raise KeyboardInterrupt
            return interrupted(), NS(language="he")
        if audio.plain:
            segments = (NS(start=s, end=e, text=t, avg_logprob=lp, no_speech_prob=ns)
                        for s, e, t, lp, ns in SEGMENTS)
        else:  # one segment per stretch of speech, named after what was spoken and when
            base = audio.offset / SR
            segments = (NS(start=s, end=e, text=f"{spoken}@{round(base + s, 2):g}",
                           avg_logprob=-0.2 if spoken == language else -1.4,  # wrong model: unsure
                           no_speech_prob=0.01)
                        for s, e, spoken in audio.heard())
        return segments, NS(language=language or "he", duration=len(audio) / SR)


class FakeContainer:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.audio_tracks = 0 if f.read(10) == b"video-only" else 1
        self.streams = NS(audio=[object()] * self.audio_tracks, video=[object()])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_av_open(path, mode="r", **kwargs):
    STATE["av_open"].append((Path(path).name, mode))
    assert mode == "r", "a source file was opened for writing"
    return FakeContainer(path)


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
        turns = STATE.get("turns") or TURNS
        if self.version == "v4":
            return NS(speaker_diarization=FakeAnnotation([(0, 1, "X")]),
                      exclusive_speaker_diarization=FakeAnnotation(turns))
        return FakeAnnotation(turns)


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
    fw.vad = _module("faster_whisper.vad", VadOptions=FakeVadOptions,
                     get_speech_timestamps=fake_speech_timestamps)
    _module("av", open=fake_av_open)
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

# A Hebrew–Russian phone call: (start, end, language).
CONVERSATION = [
    (0.5, 6.0, "he"), (6.6, 12.0, "he"),
    (12.8, 18.0, "ru"), (18.6, 24.0, "ru"),
    (24.9, 31.0, "he"),
    (31.6, 32.4, "ru"),  # a word or two in Russian
    (33.0, 38.0, "he"),
    (38.6, 44.0, "ru"),
]
CONVERSATION_RUNS = [(0.0, 12.4, "he"), (12.4, 24.45, "ru"), (24.45, 38.3, "he"), (38.3, 45.0, "ru")]


def rounded(runs):
    return [(round(start, 3), round(end, 3), language) for start, end, language in runs]


# ----------------------------------------------------------------------------
class Base(unittest.TestCase):
    def setUp(self):
        STATE.clear()
        STATE.update(pyannote="v4", hf="cached", token=None, interrupt=False, no_speech=False,
                     vad_broken=False, turns=None, vad=0, pipeline_calls=[], loaded=[],
                     detections=[], downloads=[], transcribed=[], av_open=[])
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
        self.videos = self.tmp / "videos"
        self.videos.mkdir()
        files = {"clip.mp4": b"he-video", "Lecture.MKV": b"ru-video", "song.opus": b"he-song",
                 "silent.mov": b"video-only", "clip.srt": b"1\n00:00:00,000 --> 00:00:01,000\nx\n"}
        for relative, data in files.items():
            (self.videos / relative).write_bytes(data)
        self.call = script(self.tmp / "mixed call.m4a", CONVERSATION)
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
        self.assertIs(ht.find_audio, ht.find_media)  # the 1.x name still works

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

    def test_spread(self):
        self.assertEqual(core._spread(list(range(20)), 8), [0, 3, 5, 8, 11, 14, 16, 19])
        self.assertEqual(core._spread([1, 2], 8), [1, 2])
        self.assertEqual(core._spread([], 8), [])

    def test_short_switches_join_the_surrounding_language(self):
        runs, shares = core._language_runs(
            [(0, 5), (5.5, 6.5), (7, 12), (12.5, 20)], ["he", "ru", "he", "ru"], 21.0)
        self.assertEqual(rounded(runs), [(0.0, 12.25, "he"), (12.25, 21.0, "ru")])
        self.assertAlmostEqual(shares["he"], 11 / 18.5)
        runs, shares = core._language_runs([(0, 1), (1.5, 10)], ["ru", "he"], 12.0)  # at the start
        self.assertEqual((runs, shares), ([(0.0, 12.0, "he")], {"he": 1.0}))

    def test_split_at_speaker_changes(self):
        turns = [(0.4, 10.0, "A"), (10.2, 20.0, "B"), (20.0, 25.0, "B"), (25.0, 31.0, "A")]
        self.assertEqual(core._split_at_turns([(0.5, 30.0)], turns),
                         [(0.5, 10.1), (10.1, 25.0), (25.0, 30.0)])
        self.assertEqual(core._split_at_turns([(0.5, 30.0)], None), [(0.5, 30.0)])
        # a change within MIN_TURN_PIECE of a chunk's edge does not cut it
        self.assertEqual(core._split_at_turns([(10.0, 10.2), (10.3, 15.0)], turns),
                         [(10.0, 10.2), (10.3, 15.0)])


class MediaTests(Base):
    def test_folders_contribute_audio_and_video_files(self):
        names = [p.name for p in ht.find_media([self.videos])]
        self.assertEqual(sorted(names), sorted(["Lecture.MKV", "clip.mp4", "silent.mov", "song.opus"]))
        if MEDIA.is_dir():  # the clips used by test_media.py
            fixtures = {p.name for p in ht.find_media([MEDIA])}
            self.assertEqual(fixtures, {"tone.m4a", "tone.mp4", "tone.mkv", "tone.mov", "tone.webm",
                                        "no_audio.mp4"})
        self.assertTrue({".mp4", ".mov", ".mkv", ".webm", ".avi"} <= ht.VIDEO_EXTENSIONS)
        self.assertTrue({".m4a", ".mp3", ".wav", ".opus", ".flac"} <= ht.AUDIO_EXTENSIONS)
        self.assertEqual(ht.MEDIA_EXTENSIONS, ht.AUDIO_EXTENSIONS | ht.VIDEO_EXTENSIONS)
        self.assertEqual(ht.find_media([self.videos / "clip.srt"]), [self.videos / "clip.srt"])  # explicit

    def test_a_video_is_transcribed_from_its_sound(self):
        result = ht.Transcriber(language="auto").transcribe(self.videos / "Lecture.MKV")
        self.assertEqual((result.file, result.detected_language, result.model),
                         ("Lecture.MKV", "ru", RU_TURBO))
        self.assertIn(("Lecture.MKV", "r"), STATE["av_open"])

    def test_no_sound(self):
        with self.assertRaises(ht.NoAudioError) as caught:
            ht.Transcriber().transcribe(self.videos / "silent.mov")
        self.assertIn("silent.mov has no audio track", str(caught.exception))
        self.assertIsInstance(caught.exception, ValueError)
        empty = script(self.tmp / "empty.wav", [], length=0)
        with self.assertRaises(ht.NoAudioError):
            ht.load_audio(empty)
        self.assertEqual(STATE["loaded"], [HE_TURBO])  # no transcription was attempted
        self.assertEqual(STATE["transcribed"], [])

    def test_without_pyav_the_decoder_decides(self):
        with mock.patch.dict(sys.modules, {"av": None}):
            self.assertEqual(len(ht.load_audio(self.videos / "clip.mp4")), 30 * SR)
            self.assertIsNone(core._audio_tracks(self.videos / "clip.mp4"))

    def test_cli_with_videos(self):
        code, text = self.run_cli(self.videos, "--out", self.out, "--language", "auto", "-q")
        self.assertEqual(code, 1, text)  # one file could not be transcribed
        self.assertIn("x silent.mov has no audio track", text)
        self.assertEqual({p.name for p in self.out.glob("*.txt")}, {"clip.txt", "Lecture.txt", "song.txt"})
        data = json.loads((self.out / "Lecture.json").read_text(encoding="utf-8"))
        self.assertEqual((data["source"]["file"], data["detected_language"]), ("Lecture.MKV", "ru"))
        self.assertEqual(data["source"]["sha256"], hashlib.sha256(b"ru-video").hexdigest())
        self.assertIn("Lecture.MKV", (self.out / "INDEX.md").read_text(encoding="utf-8"))

        ht.set_ui_language("ru")
        code, text = self.run_cli(self.videos / "silent.mov", "--out", self.out, "--ui-language", "ru")
        self.assertEqual(code, 1)
        self.assertIn("в файле silent.mov нет звуковой дорожки", text)


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
        self.assertEqual({s.language for s in result.segments}, {"he"})
        self.assertFalse(result.mixed)
        self.assertIsNone(result.models)
        self.assertTrue(result.text.startswith("שלום"))
        kwargs = FakeWhisperModel.last_kwargs
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertIs(kwargs["vad_filter"], True)
        self.assertEqual(kwargs["language"], "he")
        self.assertIsNone(kwargs["initial_prompt"])
        self.assertEqual(STATE["detections"], [])
        self.assertEqual(STATE["vad"], 0)  # a fixed language needs no language detection

        txt = result.save(self.out)
        self.assertEqual(txt.name, "call1.txt")
        content = txt.read_text(encoding="utf-8")
        self.assertIn(ht.UNSURE, content)
        self.assertIn("not separated", content)
        self.assertNotIn("Speaker 1:", content)
        self.assertNotIn("[he]", content)
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
            self.assertIsNone(result.languages)
            self.assertEqual(FakeWhisperModel.last_kwargs["language"], language)
            self.assertEqual(STATE["transcribed"][-1][2:], (0.0, 30.0))  # the whole file at once
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

    def test_silence_is_detected_as_a_whole(self):
        quiet = script(self.tmp / "quiet.m4a", [], length=20)
        result = ht.Transcriber(language="he,ru").transcribe(quiet)
        self.assertEqual((result.detected_language, result.segments), ("he", []))
        self.assertLess(result.language_probability, 0.5)

    def test_custom_model_detects_for_itself(self):
        result = ht.Transcriber("some/model", language="auto").transcribe(self.rec / "russian.m4a")
        self.assertEqual(STATE["loaded"], ["some/model"])
        self.assertEqual(set(STATE["detections"]), {"some/model"})
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
        for source in (self.rec / "call1.m4a", self.videos / "clip.mp4", self.call):
            before = (source.read_bytes(), source.stat().st_mtime_ns)
            ht.Transcriber(language="he,ru").transcribe(source).save(source.parent)  # even next to it
            self.assertEqual((source.read_bytes(), source.stat().st_mtime_ns), before, source.name)
        self.assertEqual({mode for _, mode in STATE["av_open"]}, {"r"})

    def test_old_transcripts_still_load(self):
        old = {"tool": "hebrew-transcribe 1.2.0", "source": {"file": "a.m4a", "sha256": "x", "bytes": 1,
               "duration_sec": 2.0}, "model": HE_TURBO, "language": "he", "detected_language": "he",
               "segments": [{"start": 0.0, "end": 1.0, "text": "שלום", "speaker": None,
                             "avg_logprob": -0.1, "no_speech_prob": 0.0, "low_confidence": False}]}
        tr = ht.Transcript.from_dict(old)
        self.assertEqual((tr.languages, tr.models, tr.segments[0].language, tr.mixed), (None, None, None, False))
        self.assertIn("שלום", output.render_txt(tr))


class MixedLanguageTests(Base):
    def test_a_conversation_in_two_languages(self):
        for setting in ("he,ru", "auto"):
            with self.subTest(language=setting):
                STATE.update(loaded=[], transcribed=[], detections=[])
                transcriber = ht.Transcriber(language=setting)
                with self.assertLogs("hebrew_transcribe", level="INFO") as logs:
                    result = transcriber.transcribe(self.call)
                self.assertTrue(result.mixed)
                self.assertEqual(result.languages, {"he": 0.5876, "ru": 0.4124})
                self.assertEqual((result.language, result.detected_language), (setting, "he"))
                self.assertIsNone(result.language_probability)
                self.assertEqual((result.model, result.models), (HE_TURBO, {"he": HE_TURBO, "ru": RU_TURBO}))
                self.assertEqual(STATE["loaded"], [DETECTOR, HE_TURBO, RU_TURBO])
                self.assertEqual(len(STATE["detections"]), 7)  # each chunk of a second or more, once
                self.assertIn("languages: Hebrew 59%, Russian 41% (mixed conversation, 4 parts)",
                              "\n".join(logs.output))

                # each stretch went to its own language's model, and times are in file time
                self.assertEqual([(name, language, start) for name, language, start, _ in STATE["transcribed"]],
                                 [(HE_TURBO, "he", 0.0), (RU_TURBO, "ru", 12.4),
                                  (HE_TURBO, "he", 24.45), (RU_TURBO, "ru", 38.3)])
                self.assertEqual([s.start for s in result.segments], [start for start, _, _ in CONVERSATION])
                self.assertEqual([s.text for s in result.segments],
                                 ["he@0.5", "he@6.6", "ru@12.8", "ru@18.6", "he@24.9", "ru@31.6",
                                  "he@33", "ru@38.6"])
                self.assertEqual([s.language for s in result.segments],
                                 ["he", "he", "ru", "ru", "he", "he", "he", "ru"])
                # the two Russian words inside Hebrew speech stay Hebrew, and are marked
                self.assertEqual([s.text for s in result.segments if s.low_confidence], ["ru@31.6"])

    def test_plan_and_detect_language(self):
        transcriber = ht.Transcriber(language="he,ru")
        audio = ht.load_audio(self.call)
        plan = transcriber.plan_languages(audio)
        self.assertIsInstance(plan, ht.LanguagePlan)
        self.assertTrue(plan.mixed)
        self.assertEqual(rounded(plan.runs), CONVERSATION_RUNS)
        self.assertEqual((plan.language, plan.probability), ("he", None))
        self.assertEqual(transcriber.detect_language(audio), ("he", 0.5876))
        fixed = ht.Transcriber(language="ru").plan_languages(audio)
        self.assertEqual((fixed.runs, fixed.mixed), ([(0.0, 45.0, "ru")], False))

    def test_files_and_index(self):
        result = ht.Transcriber(language="he,ru").transcribe(self.call)
        txt = result.save(self.out).read_text(encoding="utf-8")
        self.assertIn("he,ru → he 59%, ru 41% (mixed)", txt)
        self.assertIn(f"he: {HE_TURBO}; ru: {RU_TURBO}", txt)
        self.assertIn("[00:12] [ru] ru@12.8", txt)
        self.assertIn("[00:33] [he] he@33", txt)
        srt = (self.out / "mixed call.srt").read_text(encoding="utf-8")
        self.assertIn("00:00:12,800 --> 00:00:18,000\nru@12.8", srt)
        loaded = ht.Transcript.load(self.out / "mixed call.json")
        self.assertEqual(loaded, result)
        data = json.loads((self.out / "mixed call.json").read_text(encoding="utf-8"))
        self.assertEqual(data["languages"], {"he": 0.5876, "ru": 0.4124})
        self.assertEqual(data["segments"][2]["language"], "ru")
        ht.write_index(self.out)
        index = (self.out / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("| he 59%, ru 41% |", index)
        self.assertIn(f"| he: {HE_TURBO}; ru: {RU_TURBO} |", index)

    def test_one_odd_chunk_is_not_a_language(self):
        speech = [(0.5, 8.0, "he"), (8.6, 16.0, "he"), (16.6, 20.6, "noise"), (21.2, 29.0, "he"),
                  (29.6, 37.0, "he")]
        laugh = script(self.tmp / "laugh.m4a", speech)
        result = ht.Transcriber(language="auto").transcribe(laugh)
        self.assertFalse(result.mixed)
        self.assertEqual((result.detected_language, result.language_probability), ("he", 0.99))
        self.assertEqual(STATE["transcribed"], [(HE_TURBO, "he", 0.0, 38.0)])  # in one piece
        # Listed explicitly, a language counts even when it is heard once.
        plan = ht.Transcriber(language="he,en").plan_languages(ht.load_audio(laugh))
        self.assertEqual([language for _, _, language in plan.runs], ["he", "en", "he"])

    def test_unclear_chunks_keep_the_language_around_them(self):
        speech = [(0.5, 8.0, "he"), (8.6, 16.0, "he"), (16.6, 24.0, "ru"), (24.6, 26.6, "noise"),
                  (27.2, 35.0, "ru"), (35.6, 43.0, "he")]
        plan = ht.Transcriber(language="he,ru").plan_languages(
            ht.load_audio(script(self.tmp / "cough.m4a", speech)))
        self.assertEqual(rounded(plan.runs), [(0.0, 16.3, "he"), (16.3, 35.3, "ru"), (35.3, 44.0, "he")])

    def test_speakers_who_answer_without_a_pause(self):
        speech = [(0.5, 10.0, "he"), (10.2, 20.0, "ru"), (20.2, 30.0, "he")]  # no pause: one chunk
        fast = script(self.tmp / "fast.m4a", speech, length=31)
        alone = ht.Transcriber(language="he,ru").transcribe(fast)
        self.assertFalse(alone.mixed)  # without speaker labels the voices run together
        STATE["turns"] = [(0.4, 10.1, "SPEAKER_00"), (10.1, 20.1, "SPEAKER_01"), (20.1, 30.1, "SPEAKER_00")]
        result = ht.Transcriber(language="he,ru", diarize=True, hf_token="hf_dummy").transcribe(fast)
        self.assertTrue(result.mixed)
        self.assertEqual([(s.text, s.language, s.speaker) for s in result.segments],
                         [("he@0.5", "he", "Speaker 1"), ("ru@10.2", "ru", "Speaker 2"),
                          ("he@20.2", "he", "Speaker 1")])
        self.assertEqual(result.flagged, 0)

    def test_a_fixed_language_is_never_split(self):
        result = ht.Transcriber(language="ru").transcribe(self.call)
        self.assertEqual((STATE["vad"], STATE["detections"]), (0, []))
        self.assertEqual(STATE["transcribed"], [(RU_TURBO, "ru", 0.0, 45.0)])
        self.assertEqual({s.language for s in result.segments}, {"ru"})
        self.assertFalse(result.mixed)

    def test_when_the_voice_detector_fails(self):
        STATE["vad_broken"] = True
        with self.assertLogs("hebrew_transcribe", level="WARNING") as logs:
            result = ht.Transcriber(language="he,ru").transcribe(self.call)
        self.assertIn("could not split the audio for language detection (RuntimeError)", "\n".join(logs.output))
        self.assertEqual((result.detected_language, result.mixed), ("he", False))
        self.assertEqual(len(STATE["transcribed"]), 1)

    def test_one_language_is_only_sampled(self):
        transcriber = ht.Transcriber(language="auto")
        for count, checked in ((30, core.SCAN_CHUNKS), (200, 200 // core.SCAN_EVERY)):
            STATE["detections"] = []
            speech = [(i * 10 + 0.5, i * 10 + 9.0, "ru") for i in range(count)]
            result = transcriber.transcribe(script(self.tmp / f"long{count}.m4a", speech))
            self.assertEqual(len(STATE["detections"]), checked, count)
            self.assertEqual((result.detected_language, result.mixed), ("ru", False))

    def test_scattered_switches_are_found(self):
        # Of 30 chunks, the sample holds 12 (0, 3, 5, 8, ...): it hears Russian
        # once, in chunk 5, and so every other chunk is checked as well.
        russian = {5, 6, 7, 15}
        speech = [(i * 10 + 0.5, i * 10 + 9.0, "ru" if i in russian else "he") for i in range(30)]
        for setting in ("he,ru", "auto"):
            with self.subTest(language=setting):
                STATE["detections"] = []
                transcriber = ht.Transcriber(language=setting)
                with self.assertLogs("hebrew_transcribe", level="INFO") as logs:
                    plan = transcriber.plan_languages(ht.load_audio(script(self.tmp / "s.m4a", speech)))
                self.assertIn("checking every part of the recording", "\n".join(logs.output))
                self.assertEqual(len(STATE["detections"]), 30)  # each chunk once
                self.assertEqual([language for _, _, language in plan.runs], ["he", "ru", "he", "ru", "he"])
                self.assertEqual(rounded(plan.runs)[1], (49.75, 79.75, "ru"))
                self.assertEqual(plan.shares, {"he": 0.8667, "ru": 0.1333})

    def test_cli_mixed_in_english_and_russian(self):
        code, text = self.run_cli(self.call, "--out", self.out, "--language", "he,ru", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("languages: Hebrew 59%, Russian 41% (mixed conversation, 4 parts)", text)
        code, text = self.run_cli(self.call, "--out", self.out, "--language", "he,ru", "--force",
                                  "--ui-language", "ru", "-q")
        self.assertEqual(code, 0, text)
        self.assertIn("языки: иврит 59%, русский 41% (разговор на нескольких языках, частей: 4)", text)
        txt = (self.out / "mixed call.txt").read_text(encoding="utf-8")
        self.assertIn("he,ru → he 59%, ru 41% (несколько языков)", txt)


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
        self.assertIn("видео", out.getvalue())
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
            cli.parse_args(["x.m4a", "--language", "klingon"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
