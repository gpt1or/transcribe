"""Transcription engine: audio and video input, models per language, language
detection (also within a file, for conversations that mix languages), optional
speaker labels, and the Transcript type.

Third-party packages (faster-whisper, PyAV, huggingface_hub, pyannote.audio,
torch) are imported only when needed, so importing this module is cheap and
`transcribe --help` works before anything heavy is installed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from ._version import __version__
from .i18n import language_name, t

log = logging.getLogger("hebrew_transcribe")

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
SIZES = ("turbo", "full")

# The model used for each language: "turbo" is the default, "full" is larger
# and slower. "*" covers every other language, and its turbo model also
# detects languages.
LANGUAGE_MODELS: dict[str, dict[str, str]] = {
    "he": {  # ivrit.ai's Hebrew fine-tunes of OpenAI Whisper
        "turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",
        "full": "ivrit-ai/whisper-large-v3-ct2",
    },
    "ru": {  # Ivan Bondarenko's Russian fine-tune of Whisper large-v3-turbo (Podlodka)
        "turbo": "bond005/whisper-podlodka-turbo-ct2",
        "full": "Systran/faster-whisper-large-v3",
    },
    "*": {  # OpenAI Whisper, multilingual
        "turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
        "full": "Systran/faster-whisper-large-v3",
    },
}
DETECTION_MODEL = LANGUAGE_MODELS["*"]["turbo"]
MODELS = dict(LANGUAGE_MODELS["he"])  # the Hebrew models (the 1.x name, kept for compatibility)

DIARIZATION_MODEL_V4 = "pyannote/speaker-diarization-community-1"  # pyannote.audio 4.x
DIARIZATION_MODEL_V3 = "pyannote/speaker-diarization-3.1"  # pyannote.audio 3.x

# Language codes Whisper knows, and a few spellings people actually type.
WHISPER_LANGUAGES = frozenset(
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta "
    "no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk "
    "sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb "
    "my bo tl mg as tt haw ln ha ba jw su yue".split()
)
LANGUAGE_ALIASES = {
    "iw": "he", "heb": "he", "hebrew": "he", "иврит": "he", "עברית": "he",
    "rus": "ru", "russian": "ru", "русский": "ru", "רוסית": "ru",
    "eng": "en", "english": "en", "английский": "en", "אנגלית": "en",
}

# ----------------------------------------------------------------------------
# Input files: audio, or video (whose audio track is transcribed)
# ----------------------------------------------------------------------------
AUDIO_EXTENSIONS = frozenset({
    ".m4a", ".m4b", ".mp3", ".mp2", ".wav", ".aac", ".ac3", ".flac", ".ogg", ".oga",
    ".opus", ".wma", ".amr", ".caf", ".aif", ".aiff", ".au", ".mka", ".weba",
})
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".3gp", ".3g2", ".ogv",
})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

SAMPLE_RATE = 16_000
UNSURE = "⟨?⟩"  # the review sign: <?> in mathematical angle brackets
TOOL = f"hebrew-transcribe {__version__}"

# Segments whose average log-probability falls below this are marked UNSURE.
# -0.85 marked nothing in about 90 minutes of phone calls; -0.55 marked what
# was worth a second listen without marking the whole file.
LOW_CONFIDENCE = -0.55

# Language detection. Speech is cut into chunks at pauses (at most 30 s each)
# and, when speakers are labelled, where the speaker changes. First a sample of
# chunks spread over the whole file is checked. If it holds a second language,
# every chunk is checked, and each stretch of one language is transcribed with
# that language's model.
SCAN_CHUNKS = 12  # the sample: at least this many chunks...
SCAN_EVERY = 10  # ...or one in this many, in long recordings
MIN_SCAN_SECONDS = 2.0  # shorter chunks are not sampled
MIN_LABEL_SECONDS = 1.0  # shorter chunks, and unclear ones, keep the language before them
MIN_LANGUAGE_SECONDS = 3.0  # a listed language counts once this much of it is heard...
MIN_SHARE = 0.10  # ...and with 'auto', once it makes up this share of the speech
MIN_RUN_SECONDS = 1.5  # shorter switches (a word or two) join the surrounding language
MIN_TURN_PIECE = 0.3  # a cut where the speaker changes leaves at least this much on each side
UNCERTAIN_LANGUAGE = 0.5  # below this detection probability a language is not trusted


class ModelDownloadError(RuntimeError):
    """The speech model could not be found or downloaded."""


class DiarizationUnavailable(RuntimeError):
    """Speaker labels were requested but cannot be provided."""


class NoAudioError(ValueError):
    """The file has no audio track (for example a video without sound) or it is empty."""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def clock(seconds: float) -> str:
    """12.3 -> '00:12'; 3725 -> '1:02:05'."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def srt_time(seconds: float) -> str:
    """3.5 -> '00:00:03,500'."""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:  # read-only
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def use_system_certificates() -> bool:
    """Make Python trust the operating system's certificate store, where the
    certificate of a TLS-inspecting corporate proxy is normally installed.
    Certificate verification stays on. This affects the whole process, so the
    library never does it on its own; the command-line tool does.
    Returns False if truststore is not available."""
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


def find_media(paths: Iterable[str | os.PathLike], recursive: bool = False) -> list[Path]:
    """Audio and video files from the given files and folders, without
    duplicates. Folders contribute files with a known audio or video
    extension; files given explicitly are always included."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            entries = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                f for f in entries
                if f.is_file()
                and f.suffix.lower() in MEDIA_EXTENSIONS
                and not f.name.startswith(".")
            )
            if not candidates:
                hint = "" if recursive else t("no_audio_hint")
                log.warning(t("no_audio", path=path, hint=hint))
        elif path.is_file():
            candidates = [path]
        else:
            log.warning(t("not_found", path=path))
            continue
        for f in candidates:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                found.append(f)
    return found


find_audio = find_media  # the 1.x name


def _audio_tracks(path: Path) -> int | None:
    """How many audio tracks the file has, or None if that cannot be told."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            return len(container.streams.audio)
    except Exception:
        return None  # not readable this way: let the decoder report the problem


def load_audio(path: str | os.PathLike):
    """The sound of an audio or video file as 16 kHz mono samples. For a video
    with several audio tracks, the first one. The file is only read.
    Raises NoAudioError for a file without sound."""
    from faster_whisper.audio import decode_audio

    src = Path(path).expanduser()
    if _audio_tracks(src) == 0:
        raise NoAudioError(t("no_audio_track", file=src.name))
    audio = decode_audio(str(src), sampling_rate=SAMPLE_RATE)
    if len(audio) == 0:
        raise NoAudioError(t("empty_audio", file=src.name))
    return audio


# ----------------------------------------------------------------------------
# Languages
# ----------------------------------------------------------------------------
def parse_language(language: str | Iterable[str] | None) -> tuple[str | None, tuple[str, ...] | None]:
    """Split a language setting into (fixed language, candidates):
    'he' -> ('he', None); 'auto' or None -> (None, None), detect any language;
    'he,ru' or ['he', 'ru'] -> (None, ('he', 'ru')), detect among those.
    Raises ValueError for an unknown language code."""
    if language is None:
        return None, None
    if isinstance(language, str):
        text = language.strip().lower()
        if text in ("", "auto"):
            return None, None
        raw = [part for part in text.split(",") if part.strip()]
    else:
        raw = [str(part) for part in language]
    codes: list[str] = []
    for part in raw:
        code = LANGUAGE_ALIASES.get(part.strip().lower(), part.strip().lower())
        if code not in WHISPER_LANGUAGES:
            raise ValueError(t("error.language", code=part.strip()))
        if code not in codes:
            codes.append(code)
    if not codes:
        return None, None
    return (codes[0], None) if len(codes) == 1 else (None, tuple(codes))


def model_for(language: str | None, size: str = "turbo") -> str:
    """The model for a language code and size ('turbo' or 'full'): Hebrew and
    Russian have fine-tuned models, every other language the multilingual one."""
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, not {size!r}")
    return LANGUAGE_MODELS.get(language or "*", LANGUAGE_MODELS["*"])[size]


def required_models(model: str = "turbo", language: str | Iterable[str] | None = "he") -> list[str]:
    """Every model a run with these settings may need, e.g. to download them in
    advance. With language detection that is the detection model plus the
    model of each candidate language (Hebrew and Russian for 'auto')."""
    fixed, candidates = parse_language(language)
    if model not in SIZES:
        return [model]
    if fixed is not None:
        return [model_for(fixed, model)]
    names = [DETECTION_MODEL] + [model_for(code, model) for code in (candidates or ("he", "ru"))]
    return list(dict.fromkeys(names))


def _speech_chunks(audio: Any) -> list[tuple[float, float]]:
    """(start, end) in seconds of each stretch of speech, cut at pauses, at most 30 s each."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=200, max_speech_duration_s=30)
    return [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE)
            for c in get_speech_timestamps(audio, options)]


def _split_at_turns(chunks: list[tuple[float, float]],
                    turns: list[tuple[float, float, str]] | None) -> list[tuple[float, float]]:
    """Cut chunks of speech where the speaker changes (from speaker turns), so
    that a change of language between speakers is found even when they do not
    pause between them."""
    ordered = sorted(turns or [])
    changes = sorted((a[1] + b[0]) / 2 for a, b in zip(ordered, ordered[1:]) if a[2] != b[2])
    if not changes:
        return list(chunks)
    pieces: list[tuple[float, float]] = []
    for start, end in chunks:
        cuts = [c for c in changes if start + MIN_TURN_PIECE < c < end - MIN_TURN_PIECE]
        edges = [start, *cuts, end]
        pieces.extend(zip(edges, edges[1:]))
    return pieces


def _slice(audio: Any, start: float, end: float) -> Any:
    return audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]


def _spread(items: list, count: int) -> list:
    """Up to `count` items evenly spread over the list, first and last included."""
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def _language_runs(chunks: list[tuple[float, float]], labels: list[str],
                   duration: float) -> tuple[list[tuple[float, float, str]], dict[str, float]]:
    """Consecutive chunks of one language become one run. Very short runs join
    their neighbours. The runs cover the whole file: each boundary lies halfway
    through the pause between two languages. Returns the runs and each
    language's share of the speech."""
    groups: list[list] = []  # [language, first start, last end, seconds of speech]

    def add(group: list) -> None:
        if groups and groups[-1][0] == group[0]:
            groups[-1][2] = max(groups[-1][2], group[2])
            groups[-1][3] += group[3]
        else:
            groups.append(group)

    for (start, end), language in zip(chunks, labels):
        add([language, start, end, end - start])

    while len(groups) > 1:
        short = next((i for i, g in enumerate(groups) if g[3] < MIN_RUN_SECONDS), None)
        if short is None:
            break
        g = groups.pop(short)
        target = groups[short - 1] if short > 0 else groups[0]
        target[1], target[2] = min(target[1], g[1]), max(target[2], g[2])
        target[3] += g[3]
        merged, groups = groups, []
        for item in merged:
            add(item)

    runs = []
    for i, (language, start, end, _) in enumerate(groups):
        run_start = 0.0 if i == 0 else (groups[i - 1][2] + start) / 2
        run_end = duration if i == len(groups) - 1 else (end + groups[i + 1][1]) / 2
        runs.append((run_start, run_end, language))
    speech = sum(g[3] for g in groups) or 1.0
    shares: dict[str, float] = {}
    for language, _, _, seconds in groups:
        shares[language] = shares.get(language, 0.0) + seconds / speech
    return runs, shares


@dataclass
class LanguagePlan:
    """Which language each part of a file is in: `runs` are (start, end,
    language) and cover the whole file. `probability` is set when the language
    was detected and is the same throughout; `shares` when it changes."""

    runs: list[tuple[float, float, str]]
    language: str
    probability: float | None = None
    shares: dict[str, float] | None = None

    @property
    def mixed(self) -> bool:
        return len({language for _, _, language in self.runs}) > 1


# ----------------------------------------------------------------------------
# Downloading models
# ----------------------------------------------------------------------------
def resolve_model(name: str = "turbo", attempts: int = 4) -> str:
    """Local folder of a CTranslate2 Whisper model ('turbo' and 'full' mean the
    Hebrew models; otherwise a Hugging Face id or a local folder). Uses the copy
    on disk when there is one, without contacting the network; otherwise
    downloads it once, with retries. Raises ModelDownloadError."""
    name = MODELS.get(name, name)
    local = Path(name).expanduser()
    if local.is_dir():
        return str(local)

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(name, local_files_only=True)
    except Exception:
        pass

    log.info(t("downloading", model=name))
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # Fewer parallel connections = fewer resets by corporate proxies.
            return snapshot_download(name, max_workers=2)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            kind = type(e).__name__
            if kind in ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError"):
                raise ModelDownloadError(t("model_not_found", model=name, kind=kind)) from e
            last = e
            log.warning(t("attempt_failed", attempt=attempt, attempts=attempts,
                          error=f"{kind}: {e}"[:200]))
            if attempt < attempts:
                time.sleep(4 * attempt)
    raise ModelDownloadError(
        t("download_failed", model=name, error=f"{type(last).__name__}: {last}")
    ) from last


# ----------------------------------------------------------------------------
# Speaker labels (optional)
# ----------------------------------------------------------------------------
def default_diarization_model() -> str:
    try:
        from importlib.metadata import version

        major = int(version("pyannote.audio").split(".")[0])
    except Exception:
        major = 4
    return DIARIZATION_MODEL_V4 if major >= 4 else DIARIZATION_MODEL_V3


def find_hf_token() -> str | None:
    """A Hugging Face token from HF_TOKEN (or older variable names), or the one
    saved by `hf auth login`. Never printed or logged."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token() or None
    except Exception:
        return None


class Diarizer:
    """Tells voices apart ("Speaker 1", "Speaker 2") with pyannote.audio 3.x or
    4.x. It does not identify anyone; matching a label to a person is up to you.

    Raises DiarizationUnavailable if pyannote.audio is not installed, the model
    is not accessible (terms not accepted, missing or invalid token), or the
    pipeline would run in the cloud."""

    def __init__(self, token: str | None = None, model: str | None = None, device: str = "cpu"):
        self.model = model or default_diarization_model()
        lowered = self.model.lower()
        if "precision" in lowered or lowered.endswith("-cloud"):
            # Those pipelines run on pyannoteAI's servers: the audio would leave the machine.
            raise DiarizationUnavailable(t("diarize_cloud", model=self.model))
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationUnavailable(t("diarize_extra")) from e
        try:
            try:
                pipeline = Pipeline.from_pretrained(self.model, token=token)
            except TypeError:  # pyannote.audio 3.x named the argument differently
                pipeline = Pipeline.from_pretrained(self.model, use_auth_token=token)
        except Exception as e:
            raise DiarizationUnavailable(
                t("diarize_failed", model=self.model, kind=type(e).__name__, error=str(e)[:160])
            ) from e
        if pipeline is None:  # 3.x returns None when access is refused
            raise DiarizationUnavailable(t("diarize_no_access", model=self.model))
        if device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

    def __call__(self, audio: Any, speakers: int | None = None) -> list[tuple[float, float, str]]:
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0)  # (channel, samples), in memory
        options = {"num_speakers": speakers} if speakers else {}
        output = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **options)

        # 4.x returns a DiarizeOutput. Its "exclusive" view has no overlapping
        # turns, which maps cleanly onto transcript segments. 3.x returns the
        # annotation itself.
        annotation = None
        for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
            annotation = getattr(output, attribute, None)
            if annotation is not None:
                break
        if annotation is None:
            annotation = output

        labels: dict[str, str] = {}
        turns: list[tuple[float, float, str]] = []
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            if speaker not in labels:
                labels[speaker] = f"Speaker {len(labels) + 1}"
            turns.append((float(turn.start), float(turn.end), labels[speaker]))
        return turns


def speaker_for(start: float, end: float, turns: list[tuple[float, float, str]]) -> str | None:
    """The speaker who talks the most during [start, end]."""
    best, best_overlap = None, 0.0
    for turn_start, turn_end, label in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    return best


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    """One stretch of speech. Times are in seconds from the start of the file;
    `language` is the language it was transcribed in."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    low_confidence: bool = False
    language: str | None = None


@dataclass
class Transcript:
    """The result for one file. `save()` writes .txt, .srt and .json; `load()`
    reads a .json back. `language` is the setting asked for ('he', 'auto',
    'he,ru'); `detected_language` is the language used (the main one, if the
    conversation mixes languages); `languages` gives each language's share of
    the speech when there are several."""

    file: str
    sha256: str
    bytes: int
    duration: float
    model: str
    language: str
    detected_language: str | None = None
    language_probability: float | None = None
    languages: dict[str, float] | None = None
    models: dict[str, str] | None = None
    diarization: str | None = None
    created: str = ""
    low_confidence_threshold: float = LOW_CONFIDENCE
    processing_sec: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    tool: str = TOOL

    @property
    def text(self) -> str:
        """The whole transcript as plain text, without timestamps or labels."""
        return " ".join(s.text for s in self.segments)

    @property
    def flagged(self) -> int:
        """How many segments are marked as low confidence."""
        return sum(1 for s in self.segments if s.low_confidence)

    @property
    def mixed(self) -> bool:
        """True when the conversation was transcribed in more than one language."""
        return bool(self.languages) and len(self.languages) > 1

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "source": {
                "file": self.file,
                "sha256": self.sha256,
                "bytes": self.bytes,
                "duration_sec": self.duration,
            },
            "model": self.model,
            "models": self.models,
            "language": self.language,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
            "languages": self.languages,
            "diarization": self.diarization,
            "created": self.created,
            "low_confidence_threshold": self.low_confidence_threshold,
            "processing_sec": self.processing_sec,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        source = data.get("source", {})
        known = {f for f in Segment.__dataclass_fields__}
        return cls(
            file=source.get("file", ""),
            sha256=source.get("sha256", ""),
            bytes=source.get("bytes", 0),
            duration=source.get("duration_sec", 0.0),
            model=data.get("model", ""),
            models=data.get("models"),
            language=data.get("language", ""),
            detected_language=data.get("detected_language"),
            language_probability=data.get("language_probability"),
            languages=data.get("languages"),
            diarization=data.get("diarization"),
            created=data.get("created", ""),
            low_confidence_threshold=data.get("low_confidence_threshold", LOW_CONFIDENCE),
            processing_sec=data.get("processing_sec", 0.0),
            segments=[Segment(**{k: v for k, v in s.items() if k in known})
                      for s in data.get("segments", [])],
            tool=data.get("tool", TOOL),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def to_srt(self) -> str:
        from .output import render_srt

        return render_srt(self)

    def to_txt(self) -> str:
        from .output import render_txt

        return render_txt(self)

    def save(self, out_dir: str | os.PathLike, name: str | None = None) -> Path:
        """Write <name>.json, <name>.srt and <name>.txt (default name: the source
        file name without its extension). Returns the path of the .txt file."""
        from .output import save

        return save(self, out_dir, name)


# ----------------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------------
class Transcriber:
    """Loads the models once; `transcribe()` then handles any number of audio
    or video files.

    model      'turbo' (default) or 'full', which pick the model that suits the
               language (see LANGUAGE_MODELS), or one fixed model: a Hugging Face
               id or a local folder
    language   'he' (default), 'ru' or any other code; 'auto' (or None) to detect
               the language, also when a conversation mixes languages; or a list
               such as 'he,ru' to choose among those languages only. With
               speaker labels, a change of language between speakers is found
               even when they do not pause
    device     'auto' (an NVIDIA GPU if available, else the CPU), 'cpu', 'cuda'
    diarize    True to add speaker labels (needs the [diarize] extra and a
               Hugging Face token: hf_token=, HF_TOKEN or `hf auth login`)
    diarizer   an existing Diarizer to use instead of creating one
    prompt     names and terms that occur in the audio, to help recognise them
    flag_below segments below this average log-probability are marked low-confidence

    Models are loaded when first needed and then kept. Raises ValueError for an
    unknown language, ModelDownloadError, or DiarizationUnavailable when
    diarize=True and speaker labels cannot be provided.
    """

    def __init__(
        self,
        model: str = "turbo",
        *,
        language: str | Iterable[str] | None = "he",
        device: str = "auto",
        threads: int | None = None,
        diarize: bool = False,
        hf_token: str | None = None,
        diarization_model: str | None = None,
        diarizer: Diarizer | None = None,
        prompt: str | None = None,
        flag_below: float = LOW_CONFIDENCE,
    ):
        self.size = model if model in SIZES else None
        self.custom_model = None if self.size else model
        self.fixed_language, self.candidates = parse_language(language)
        self.language = self.fixed_language  # None: detected for each file
        if self.fixed_language:
            self.requested_language = self.fixed_language
        else:
            self.requested_language = ",".join(self.candidates) if self.candidates else "auto"
        self.device = ("cuda" if cuda_available() else "cpu") if device == "auto" else device
        self.threads = threads or os.cpu_count() or 4
        self.prompt = prompt
        self.flag_below = flag_below
        if diarizer is None and diarize:
            diarizer = Diarizer(hf_token or find_hf_token(), diarization_model, self.device)
        self.diarizer = diarizer

        self._loaded: dict[str, Any] = {}
        if self.fixed_language is None:
            # Detection: a custom model detects for itself; otherwise the multilingual model.
            self.detector_name: str | None = self.custom_model or DETECTION_MODEL
            self.model_name = self.detector_name
        else:
            self.detector_name = None
            self.model_name = self.custom_model or model_for(self.fixed_language, self.size)
        self.model = self._load(self.model_name)

    def _load(self, name: str):
        """The WhisperModel for `name`, loading (and downloading) it on first use."""
        if name not in self._loaded:
            if self._loaded:  # the first model is announced by the caller
                log.info(t("loading_model", model=name))
            model_dir = resolve_model(name)
            from faster_whisper import WhisperModel

            self._loaded[name] = WhisperModel(
                model_dir,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
                cpu_threads=self.threads,
            )
        return self._loaded[name]

    def _model_name_for(self, language: str) -> str:
        return self.custom_model or model_for(language, self.size)

    def _scores(self, audio: Any) -> dict[str, float]:
        """Every language's probability for a piece of audio."""
        detector = self._load(self.detector_name or DETECTION_MODEL)
        language, probability, scores = detector.detect_language(
            audio, vad_filter=False, language_detection_segments=1, language_detection_threshold=0.5,
        )
        table = {code: float(p) for code, p in (scores or ())}
        table.setdefault(language, float(probability))
        return table

    @staticmethod
    def _best(table: dict[str, float], allowed: Iterable[str] | None = None) -> tuple[str, float]:
        """The most probable language in `table`, among `allowed` if given."""
        language = max(list(allowed or table), key=lambda code: table.get(code, 0.0))
        return language, table.get(language, 0.0)

    def _identify(self, audio: Any, allowed: Iterable[str] | None) -> tuple[str, float]:
        """(language, probability) of a piece of audio, limited to `allowed` if given."""
        return self._best(self._scores(audio), allowed)

    def plan_languages(self, audio: Any,
                       turns: list[tuple[float, float, str]] | None = None) -> LanguagePlan:
        """Which language each part of decoded 16 kHz audio is in. With speaker
        turns (from a Diarizer), a change of language between speakers is found
        even when they do not pause."""
        duration = len(audio) / SAMPLE_RATE
        if self.fixed_language:
            return LanguagePlan([(0.0, duration, self.fixed_language)], self.fixed_language)
        fallback = self.candidates[0] if self.candidates else "he"

        def whole(language: str, probability: float | None) -> LanguagePlan:
            return LanguagePlan([(0.0, duration, language)], language, probability)

        try:
            chunks = _split_at_turns(_speech_chunks(audio), turns)
        except Exception as e:  # e.g. the voice detector cannot run here
            log.warning(t("scan_failed", kind=type(e).__name__))
            chunks = []
        tables: dict[int, dict[str, float]] = {}  # chunk -> scores; no chunk is checked twice

        def check(indices: Iterable[int]) -> dict[str, list[tuple[float, float]]]:
            """language -> [(seconds, probability)] for these chunks."""
            heard: dict[str, list[tuple[float, float]]] = {}
            for i in indices:
                try:
                    if i not in tables:
                        tables[i] = self._scores(_slice(audio, *chunks[i]))
                    language, probability = self._best(tables[i], self.candidates)
                except Exception:
                    continue
                heard.setdefault(language, []).append((chunks[i][1] - chunks[i][0], probability))
            return heard

        def confident(heard: dict) -> dict[str, list[float]]:
            """Seconds of each language heard with confidence, the most heard first."""
            seconds = {code: [s for s, p in items if p >= UNCERTAIN_LANGUAGE] for code, items in heard.items()}
            return dict(sorted(((code, s) for code, s in seconds.items() if s), key=lambda kv: -sum(kv[1])))

        def confidence(heard: dict, code: str) -> float:
            items = heard.get(code, [])
            return sum(s * p for s, p in items) / (sum(s for s, _ in items) or 1.0)

        # 1. A sample of the speech, spread over the whole file.
        pool = [i for i, (start, end) in enumerate(chunks) if end - start >= MIN_SCAN_SECONDS]
        pool = pool or list(range(len(chunks)))
        heard = check(_spread(pool, max(SCAN_CHUNKS, len(pool) // SCAN_EVERY)))
        if not heard:  # no speech found, or no part could be checked: try the whole file
            try:
                return whole(*self._identify(audio, self.candidates))
            except Exception:
                log.warning(t("no_speech_detected", language=language_name(fallback)))
                return whole(fallback, 0.0)
        sure = confident(heard)
        main = next(iter(sure), None) or max(heard, key=lambda code: sum(s * p for s, p in heard[code]))
        if len(sure) < 2:  # one language throughout
            return whole(main, confidence(heard, main))

        # 2. A second language was heard: check every part of the speech. A
        # listed language counts once a few seconds of it are heard. With
        # 'auto' any language is possible, so it must be heard in two places and
        # make up a real part of the speech: one odd chunk (a laugh) is not enough.
        log.info(t("checking_parts"))
        heard = check(i for i, (start, end) in enumerate(chunks) if end - start >= MIN_LABEL_SECONDS)
        sure = confident(heard) or {main: [0.0]}
        total = sum(sum(seconds) for seconds in sure.values()) or 1.0
        present = [code for n, (code, seconds) in enumerate(sure.items())
                   if n == 0 or (sum(seconds) >= MIN_LANGUAGE_SECONDS if self.candidates
                                 else len(seconds) >= 2 and sum(seconds) / total >= MIN_SHARE)]
        if len(present) < 2:
            return whole(present[0], confidence(heard, present[0]))

        # Each chunk keeps the language before it unless its own is clear.
        labels: list[str] = []
        for i, (start, end) in enumerate(chunks):
            label = ""
            if i in tables and end - start >= MIN_LABEL_SECONDS:
                language, probability = self._best(tables[i], present)
                if probability >= UNCERTAIN_LANGUAGE:
                    label = language
            labels.append(label or (labels[-1] if labels else ""))
        first = next((label for label in labels if label), present[0])
        labels = [label or first for label in labels]
        runs, shares = _language_runs(chunks, labels, duration)
        if len(shares) < 2:
            return whole(runs[0][2], confidence(heard, runs[0][2]))
        main = max(shares, key=shares.get)
        return LanguagePlan(runs, main, None, {code: round(share, 4) for code, share in shares.items()})

    def detect_language(self, audio: Any) -> tuple[str, float]:
        """(main language, probability) for decoded 16 kHz audio. For a
        conversation in several languages the probability is the main
        language's share of the speech."""
        plan = self.plan_languages(audio)
        if plan.shares:
            return plan.language, plan.shares[plan.language]
        return plan.language, plan.probability if plan.probability is not None else 1.0

    def transcribe(
        self,
        path: str | os.PathLike,
        *,
        speakers: int | None = None,
        prompt: str | None = None,
        on_segment: Callable[[Segment], None] | None = None,
    ) -> Transcript:
        """Transcribe one audio or video file. The file is only read.

        speakers    number of speakers, if known (with speaker labels)
        prompt      overrides the prompt given to the Transcriber
        on_segment  called with each Segment as soon as it is ready

        Raises NoAudioError for a file without sound.
        """
        src = Path(path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(src)
        started = time.time()
        digest = sha256_of(src)
        audio = load_audio(src)
        duration = len(audio) / SAMPLE_RATE

        turns: list[tuple[float, float, str]] = []
        if self.diarizer is not None:
            log.info(t("separating"))
            try:
                turns = self.diarizer(audio, speakers or None)
            except Exception as e:
                log.warning(t("separation_failed", kind=type(e).__name__))

        plan = self.plan_languages(audio, turns)
        if plan.shares:
            names = ", ".join(f"{language_name(code)} {round(share * 100)}%"
                              for code, share in sorted(plan.shares.items(), key=lambda kv: -kv[1]))
            log.info(t("mixed", languages=names, parts=len(plan.runs)))
        elif plan.probability is not None:
            log.info(t("detected", language=language_name(plan.language),
                       percent=round(plan.probability * 100), model=self._model_name_for(plan.language)))
            if plan.probability < UNCERTAIN_LANGUAGE:
                log.warning(t("uncertain"))

        segments: list[Segment] = []
        used: dict[str, str] = {}
        for run_start, run_end, language in plan.runs:
            name = self._model_name_for(language)
            used[language] = name
            if len(plan.runs) == 1:
                piece, offset = audio, 0.0
            else:  # this stretch only; its times are then shifted back into place
                first, last = int(run_start * SAMPLE_RATE), int(run_end * SAMPLE_RATE)
                piece, offset = audio[first:last], first / SAMPLE_RATE
            raw_segments, _info = self._load(name).transcribe(
                piece,
                language=language,
                beam_size=5,
                vad_filter=True,  # skip silence: fewer invented words
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,  # prevents repetition loops
                initial_prompt=(self.prompt if prompt is None else prompt) or None,
                temperature=[0.0, 0.2, 0.4],
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            for seg in raw_segments:  # transcription happens while iterating
                text = seg.text.strip()
                if not text:
                    continue
                start, end = float(seg.start) + offset, float(seg.end) + offset
                logprob = getattr(seg, "avg_logprob", None)
                no_speech = getattr(seg, "no_speech_prob", None)
                segment = Segment(
                    start=round(start, 3),
                    end=round(end, 3),
                    text=text,
                    speaker=speaker_for(start, end, turns),
                    avg_logprob=None if logprob is None else round(float(logprob), 4),
                    no_speech_prob=None if no_speech is None else round(float(no_speech), 4),
                    low_confidence=logprob is not None and logprob < self.flag_below,
                    language=language,
                )
                segments.append(segment)
                if on_segment is not None:
                    on_segment(segment)

        return Transcript(
            file=src.name,
            sha256=digest,
            bytes=src.stat().st_size,
            duration=round(duration, 2),
            model=used.get(plan.language) or next(iter(used.values())),
            models=used if len(used) > 1 else None,
            language=self.requested_language,
            detected_language=plan.language,
            language_probability=None if plan.probability is None else round(plan.probability, 4),
            languages=plan.shares,
            diarization=self.diarizer.model if (self.diarizer is not None and turns) else None,
            created=datetime.now().astimezone().isoformat(timespec="seconds"),
            low_confidence_threshold=self.flag_below,
            processing_sec=round(time.time() - started, 1),
            segments=segments,
        )


def transcribe(
    path: str | os.PathLike,
    *,
    model: str = "turbo",
    language: str | Iterable[str] | None = "he",
    diarize: bool = False,
    speakers: int | None = None,
    prompt: str | None = None,
    device: str = "auto",
    hf_token: str | None = None,
) -> Transcript:
    """Transcribe one audio or video file in one call. For several files,
    create a Transcriber once and reuse it: loading the models is the slow part."""
    transcriber = Transcriber(
        model, language=language, device=device, diarize=diarize, hf_token=hf_token, prompt=prompt
    )
    return transcriber.transcribe(path, speakers=speakers)
