"""Transcription engine: models per language, language detection, optional
speaker labels, and the Transcript type.

Third-party packages (faster-whisper, huggingface_hub, pyannote.audio, torch)
are imported only when needed, so importing this module is cheap and
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
# detects the language with --language auto.
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

AUDIO_EXTENSIONS = frozenset({
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".amr", ".3gp", ".caf", ".aif", ".aiff", ".webm", ".mp4", ".m4v", ".mov",
    ".mkv",
})

SAMPLE_RATE = 16_000
UNSURE = "⟨?⟩"  # the review sign: <?> in mathematical angle brackets
TOOL = f"hebrew-transcribe {__version__}"

# Segments whose average log-probability falls below this are marked UNSURE.
# -0.85 marked nothing in about 90 minutes of phone calls; -0.55 marked what
# was worth a second listen without marking the whole file.
LOW_CONFIDENCE = -0.55

# Language detection: up to 3 windows of 30 s of speech; below this
# probability the result is reported as uncertain.
UNCERTAIN_LANGUAGE = 0.5


class ModelDownloadError(RuntimeError):
    """The speech model could not be found or downloaded."""


class DiarizationUnavailable(RuntimeError):
    """Speaker labels were requested but cannot be provided."""


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


def find_audio(paths: Iterable[str | os.PathLike], recursive: bool = False) -> list[Path]:
    """Audio/video files from the given files and folders, without duplicates.
    Folders contribute files with a known audio/video extension; files given
    explicitly are always included."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            entries = path.rglob("*") if recursive else path.iterdir()
            candidates = sorted(
                f for f in entries
                if f.is_file()
                and f.suffix.lower() in AUDIO_EXTENSIONS
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
    """One stretch of speech. Times are in seconds from the start of the file."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    low_confidence: bool = False


@dataclass
class Transcript:
    """The result for one file. `save()` writes .txt, .srt and .json; `load()`
    reads a .json back. `language` is the setting asked for ('he', 'auto',
    'he,ru'); `detected_language` is the language actually used."""

    file: str
    sha256: str
    bytes: int
    duration: float
    model: str
    language: str
    detected_language: str | None = None
    language_probability: float | None = None
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
            "language": self.language,
            "detected_language": self.detected_language,
            "language_probability": self.language_probability,
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
            language=data.get("language", ""),
            detected_language=data.get("detected_language"),
            language_probability=data.get("language_probability"),
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
    """Loads the models once; `transcribe()` then handles any number of files.

    model      'turbo' (default) or 'full', which pick the model that suits the
               language (see LANGUAGE_MODELS), or one fixed model: a Hugging Face
               id or a local folder
    language   'he' (default), 'ru' or any other code; 'auto' (or None) to detect
               each file's language and use the matching model; or a list such as
               'he,ru' to choose among those languages only
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

    def detect_language(self, audio: Any) -> tuple[str, float]:
        """(language code, probability) for decoded 16 kHz audio, limited to the
        candidate languages if any were given."""
        detector = self._load(self.detector_name or DETECTION_MODEL)
        fallback = self.candidates[0] if self.candidates else "he"
        result = None
        for use_vad in (True, False):  # without VAD if it finds no speech at all
            try:
                result = detector.detect_language(
                    audio, vad_filter=use_vad,
                    language_detection_segments=3, language_detection_threshold=0.5,
                )
                break
            except Exception:
                continue
        if result is None:
            log.warning(t("no_speech_detected", language=language_name(fallback)))
            return fallback, 0.0
        language, probability, all_probabilities = result
        if self.candidates:
            scores = dict(all_probabilities)
            language = max(self.candidates, key=lambda code: scores.get(code, 0.0))
            probability = scores.get(language, 0.0)
        return language, float(probability)

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
        """
        from faster_whisper.audio import decode_audio

        src = Path(path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(src)
        started = time.time()
        digest = sha256_of(src)
        audio = decode_audio(str(src), sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE

        turns: list[tuple[float, float, str]] = []
        if self.diarizer is not None:
            log.info(t("separating"))
            try:
                turns = self.diarizer(audio, speakers or None)
            except Exception as e:
                log.warning(t("separation_failed", kind=type(e).__name__))

        language, probability = self.fixed_language, None
        if language is None:
            language, probability = self.detect_language(audio)
        model_name = self.custom_model or model_for(language, self.size)
        if probability is not None:
            log.info(t("detected", language=language_name(language),
                       percent=round(probability * 100), model=model_name))
            if probability < UNCERTAIN_LANGUAGE:
                log.warning(t("uncertain"))
        model = self._load(model_name)

        raw_segments, info = model.transcribe(
            audio,
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

        segments: list[Segment] = []
        for seg in raw_segments:  # transcription happens while iterating
            text = seg.text.strip()
            if not text:
                continue
            logprob = getattr(seg, "avg_logprob", None)
            no_speech = getattr(seg, "no_speech_prob", None)
            segment = Segment(
                start=round(float(seg.start), 3),
                end=round(float(seg.end), 3),
                text=text,
                speaker=speaker_for(seg.start, seg.end, turns),
                avg_logprob=None if logprob is None else round(float(logprob), 4),
                no_speech_prob=None if no_speech is None else round(float(no_speech), 4),
                low_confidence=logprob is not None and logprob < self.flag_below,
            )
            segments.append(segment)
            if on_segment is not None:
                on_segment(segment)

        return Transcript(
            file=src.name,
            sha256=digest,
            bytes=src.stat().st_size,
            duration=round(duration, 2),
            model=model_name,
            language=self.requested_language,
            detected_language=language or getattr(info, "language", None),
            language_probability=None if probability is None else round(probability, 4),
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
    """Transcribe one file in one call. For several files, create a Transcriber
    once and reuse it: loading the models is the slow part."""
    transcriber = Transcriber(
        model, language=language, device=device, diarize=diarize, hf_token=hf_token, prompt=prompt
    )
    return transcriber.transcribe(path, speakers=speakers)
