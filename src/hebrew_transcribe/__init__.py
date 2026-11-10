"""hebrew-transcribe: private, local Hebrew speech-to-text with optional speaker labels.

    from hebrew_transcribe import Transcriber

    transcriber = Transcriber()                  # loads (and once downloads) the model
    result = transcriber.transcribe("call.m4a")
    print(result.text)
    result.save("transcripts")                   # .txt, .srt and .json

Everything runs locally; audio, transcripts and tokens are never sent anywhere.
Machine transcription makes mistakes: verify every transcript against the
original audio before relying on it. See the Disclaimer in the README.
"""
import os as _os

# Privacy defaults, applied before any third-party package is imported.
# pyannote.audio 4.x sends usage metrics by default and huggingface_hub reports
# library usage. Neither receives audio, but this package promises that nothing
# leaves the machine. setdefault: an explicit choice by the caller always wins.
_os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from ._version import __version__  # noqa: E402
from .core import (  # noqa: E402
    LOW_CONFIDENCE,
    MODELS,
    UNSURE,
    DiarizationUnavailable,
    Diarizer,
    ModelDownloadError,
    Segment,
    Transcriber,
    Transcript,
    find_audio,
    resolve_model,
    transcribe,
    use_system_certificates,
)
from .output import write_index  # noqa: E402

__all__ = [
    "__version__",
    "LOW_CONFIDENCE",
    "MODELS",
    "UNSURE",
    "DiarizationUnavailable",
    "Diarizer",
    "ModelDownloadError",
    "Segment",
    "Transcriber",
    "Transcript",
    "find_audio",
    "resolve_model",
    "transcribe",
    "use_system_certificates",
    "write_index",
]
