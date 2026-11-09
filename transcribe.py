#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe - private, local Hebrew speech-to-text with optional speaker labels.

  engine    faster-whisper (CTranslate2)
  model     ivrit-ai/whisper-large-v3-turbo-ct2, ivrit.ai's Hebrew fine-tune
            of OpenAI Whisper large-v3-turbo
  speakers  pyannote.audio (optional, off by default)

Everything runs on your computer. The network is used only to download the
models once; after that the transcriber works offline. This program never
sends your audio, transcripts or tokens anywhere, and the usage telemetry of
the libraries it uses is switched off.

For every input file, the output folder receives:
  <name>.txt   readable transcript with timestamps (and speakers, if enabled)
  <name>.srt   subtitles, for checking the text against the audio in a player
  <name>.json  segments with timings and confidence scores, for further use
plus INDEX.md, a table of everything transcribed into that folder so far.

Deliberate choices (see README):
  * condition_on_previous_text=False prevents Whisper's repetition loops and
    invented text, its classic failure on phone calls with silence or noise.
  * Voice-activity detection skips silence, for the same reason.
  * Low-confidence segments are marked with a review sign for a human check.
  * Source files are only read: never modified, moved or converted. Each
    transcript records the SHA-256 of its source file, so it can be matched
    to the exact file it came from.

Usage:  python transcribe.py <files or folders> [options]     (--help for all)

Machine transcription makes mistakes. Verify every transcript against the
original audio before relying on it. See the Disclaimer in README.md.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

__version__ = "1.0.0"


def _prepare_environment() -> None:
    """Runs before any third-party package is imported.

    * Telemetry off. pyannote.audio 4.x sends usage metrics by default and
      huggingface_hub reports library usage. Neither receives audio, but this
      tool promises that nothing leaves the machine. To opt back in, export
      PYANNOTE_METRICS_ENABLED=1 or HF_HUB_DISABLE_TELEMETRY=0.
    * Corporate networks. truststore makes Python trust the operating system's
      certificate store, where the certificate of a TLS-inspecting proxy is
      normally installed. Certificate verification stays ON.
    * Hugging Face's newer "Xet" download path uses a separate CDN that some
      proxies block, so the classic path is used (HF_HUB_DISABLE_XET=0 to undo).
    """
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


_prepare_environment()

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
TOKEN_FILE = APP_DIR / ".hf_token"  # git-ignored, readable only by you

MODELS = {
    "turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",  # default: fast and accurate
    "full": "ivrit-ai/whisper-large-v3-ct2",  # about 3x slower, sometimes better
}
DIARIZATION_MODEL_V4 = "pyannote/speaker-diarization-community-1"  # pyannote.audio 4.x
DIARIZATION_MODEL_V3 = "pyannote/speaker-diarization-3.1"  # pyannote.audio 3.x

AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".amr", ".3gp", ".caf", ".aif", ".aiff", ".webm", ".mp4", ".m4v", ".mov",
    ".mkv",
}

SAMPLE_RATE = 16_000
UNSURE = "⟨?⟩"  # the review sign: <?> in mathematical angle brackets

# Segments whose average log-probability falls below this are marked UNSURE.
# -0.85 marked nothing in about 90 minutes of phone calls; -0.55 marked what
# was worth a second listen without marking the whole file.
LOW_CONFIDENCE = -0.55

TOKEN_HELP = """
Speaker labels use pyannote's models from Hugging Face. They are free but
gated, so a free access token is needed once:
  1. Create an account:        https://huggingface.co/join
  2. Accept the model terms:   https://huggingface.co/pyannote/speaker-diarization-community-1
  3. Create a "Read" token:    https://huggingface.co/settings/tokens
"""

NETWORK_HELP = """
  The download failed. Internet is needed only for this one-time download.
  Try, in order:
    1. Disconnect from a work VPN, or switch network (a phone hotspot works),
       then run the same command again. The download resumes where it stopped.
    2. macOS behind a corporate proxy:  bash setup.sh --corp-certs
"""


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
    with path.open("rb") as f:  # read-only
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so an interrupted run never leaves a
    half-written transcript that a later run would mistake for a finished one."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def md_cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Inputs and output names
# ----------------------------------------------------------------------------
def find_inputs(paths: list[str], recursive: bool) -> list[Path]:
    """Audio/video files from the given files and folders, without duplicates."""
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
                hint = "" if recursive else " (add --recursive for sub-folders)"
                print(f"  ! no audio files in {path}{hint}")
        elif path.is_file():
            candidates = [path]
        else:
            print(f"  x not found: {path}")
            continue
        for f in candidates:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                found.append(f)
    return found


def output_names(files: list[Path]) -> dict[Path, str]:
    """A stable, unique base name per input: the file name without its
    extension, prefixed with the folder name when two inputs share a name."""
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(f.stem, []).append(f)
    names: dict[Path, str] = {}
    used: set[str] = set()
    for stem, group in groups.items():
        for f in sorted(group):
            name = stem if len(group) == 1 else f"{f.parent.name}__{stem}"
            base, n = name, 2
            while name in used:
                name, n = f"{base}__{n}", n + 1
            used.add(name)
            names[f] = name
    return names


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def resolve_model(name: str, attempts: int = 4) -> str:
    """Local folder of a CTranslate2 Whisper model. Uses the copy on disk when
    there is one (no network at all); otherwise downloads it once, with retries."""
    local = Path(name).expanduser()
    if local.is_dir():
        return str(local)

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(name, local_files_only=True)
    except Exception:
        pass

    print(f"Downloading {name} (one time; resumes if interrupted)...", flush=True)
    for attempt in range(1, attempts + 1):
        try:
            # Fewer parallel connections = fewer resets by corporate proxies.
            return snapshot_download(name, max_workers=2)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            kind = type(e).__name__
            if kind in ("RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError"):
                sys.exit(f"x Model '{name}' was not found or is not accessible ({kind}).")
            message = f"{kind}: {e}"
            print(f"  attempt {attempt}/{attempts} failed: {message[:200]}")
            if attempt == attempts:
                print(NETWORK_HELP)
                if "403" in message or "Forbidden" in message:
                    print("  (403 Forbidden usually means a proxy or firewall blocks the download.)")
                sys.exit(1)
            time.sleep(4 * attempt)
    raise AssertionError("unreachable")


def load_whisper(model_dir: str, device: str, threads: int):
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_dir, device=device, compute_type=compute_type, cpu_threads=threads)


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


def find_token() -> str | None:
    """A Hugging Face token from the environment, from .hf_token, or from an
    earlier `hf auth login`. Never printed."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    if TOKEN_FILE.is_file():
        value = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token() or None
    except Exception:
        return None


def ask_token() -> str | None:
    """One-time interactive prompt. The input is hidden, and the token is
    stored only in .hf_token next to this script (readable by you alone)."""
    if not sys.stdin.isatty():
        return None
    print(TOKEN_HELP)
    try:
        token = getpass.getpass("Hugging Face token (hidden; Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not token:
        return None
    try:
        fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except OSError:
            pass
        print(f"  saved to {TOKEN_FILE.name} (readable only by you; excluded from git)")
    except OSError as e:
        print(f"  ! could not save the token ({e}); using it for this run only")
    return token


class Diarizer:
    """Tells voices apart ("Speaker 1", "Speaker 2") with pyannote.audio 3.x or
    4.x. It does not identify anyone; matching a label to a person is up to you."""

    def __init__(self, token: str | None, model: str | None, device: str):
        from pyannote.audio import Pipeline

        self.model = model or default_diarization_model()
        lowered = self.model.lower()
        if "precision" in lowered or lowered.endswith("-cloud"):
            # Those pipelines run on pyannoteAI's servers: the audio would leave the machine.
            raise ValueError(f"{self.model} is a cloud pipeline; only local pipelines are supported")
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token)
        except TypeError:  # pyannote.audio 3.x named the argument differently
            pipeline = Pipeline.from_pretrained(self.model, use_auth_token=token)
        if pipeline is None:  # 3.x returns None when access is refused
            raise PermissionError(f"no access to {self.model}")
        if device == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

    def __call__(self, audio, speakers: int | None) -> list[tuple[float, float, str]]:
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


def load_diarizer(args, device: str) -> Diarizer | None:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        print("! Speaker labels need extra packages:  bash setup.sh --diarize")
        print("  (or: pip install -r requirements-diarize.txt)")
        print("  Continuing without speaker labels.\n")
        return None
    token = find_token() or ask_token()
    print("Loading the speaker-separation model...", flush=True)
    try:
        diarizer = Diarizer(token, args.diarization_model, device)
    except ValueError as e:  # refused on purpose, e.g. a cloud pipeline
        print(f"! {e}.\n  Continuing without speaker labels.\n")
        return None
    except Exception as e:
        name = args.diarization_model or default_diarization_model()
        print(f"! Could not load {name} ({type(e).__name__}: {str(e)[:160]}).")
        print("  Accept the model's terms on Hugging Face with the account that owns")
        print("  the token, and check the token. See README, 'Speaker labels'.")
        print("  Continuing without speaker labels.\n")
        return None
    print(f"  ready: {diarizer.model}\n")
    return diarizer


# ----------------------------------------------------------------------------
# Transcription and output files
# ----------------------------------------------------------------------------
def transcribe_file(src: Path, model, diarizer: Diarizer | None, args) -> dict:
    from faster_whisper.audio import decode_audio

    started = time.time()
    digest = sha256_of(src)
    audio = decode_audio(str(src), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    turns: list[tuple[float, float, str]] = []
    if diarizer is not None:
        print("   separating speakers...", flush=True)
        try:
            turns = diarizer(audio, args.speakers or None)
        except Exception as e:
            print(f"   ! speaker separation failed ({type(e).__name__}); continuing without it")

    segments, info = model.transcribe(
        audio,
        language=args.language,  # None = detect
        beam_size=5,
        vad_filter=True,  # skip silence: fewer invented words
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # prevents repetition loops
        initial_prompt=args.prompt or None,
        temperature=[0.0, 0.2, 0.4],
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    rows = []
    for seg in segments:  # transcription happens while iterating
        text = seg.text.strip()
        if not text:
            continue
        logprob = getattr(seg, "avg_logprob", None)
        no_speech = getattr(seg, "no_speech_prob", None)
        rows.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "speaker": speaker_for(seg.start, seg.end, turns),
            "text": text,
            "avg_logprob": None if logprob is None else round(float(logprob), 4),
            "no_speech_prob": None if no_speech is None else round(float(no_speech), 4),
            "low_confidence": logprob is not None and logprob < args.flag_below,
        })
        if not args.quiet:
            print(f"      {clock(seg.start)}  {text[:70]}", flush=True)

    return {
        "tool": f"transcribe {__version__}",
        "source": {
            "file": src.name,
            "sha256": digest,
            "bytes": src.stat().st_size,
            "duration_sec": round(duration, 2),
        },
        "model": args.model_name,
        "language": args.language or "auto",
        "detected_language": getattr(info, "language", None),
        "diarization": diarizer.model if (diarizer is not None and turns) else None,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "low_confidence_threshold": args.flag_below,
        "processing_sec": round(time.time() - started, 1),
        "segments": rows,
    }


def render_txt(meta: dict) -> str:
    source, rows = meta["source"], meta["segments"]
    flagged = sum(1 for r in rows if r["low_confidence"])
    line = "=" * 66
    header = [
        f"Transcript - {source['file']}",
        line,
        f"Source file : {source['file']}",
        f"SHA-256     : {source['sha256']}",
        f"Duration    : {clock(source['duration_sec'])}",
        f"Model       : {meta['model']}",
        f"Language    : {meta['language']}",
        f"Speakers    : {meta['diarization'] or 'not separated'}",
        f"Created     : {meta['created']}, locally, by {meta['tool']}",
        "-" * 66,
        "Machine-generated. It may contain wrong, missing or invented words",
        f"and wrong speaker labels. {UNSURE} = low confidence. Verify against the",
        "original audio before relying on any part of it.",
        line,
    ]
    body: list[str] = []
    previous = object()
    for r in rows:
        if meta["diarization"] and r["speaker"] != previous:
            if body:
                body.append("")
            body.append(f"{r['speaker'] or 'Speaker ?'}:")
            previous = r["speaker"]
        mark = f" {UNSURE}" if r["low_confidence"] else ""
        body.append(f"  [{clock(r['start'])}] {r['text']}{mark}")
    footer = [line, f"End of transcript. {len(rows)} segments, {flagged} marked {UNSURE}.", ""]
    return "\n".join(header + [""] + body + [""] + footer)


def render_srt(rows: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(rows, 1):
        prefix = f"{r['speaker']}: " if r["speaker"] else ""
        blocks.append(f"{i}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{prefix}{r['text']}\n")
    return "\n".join(blocks)


def write_outputs(meta: dict, out_dir: Path, name: str) -> None:
    # The .txt is written last: its presence is what marks a file as done.
    write_atomic(out_dir / f"{name}.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    write_atomic(out_dir / f"{name}.srt", render_srt(meta["segments"]))
    write_atomic(out_dir / f"{name}.txt", render_txt(meta))


def write_index(out_dir: Path) -> int:
    """INDEX.md for every transcript in the folder, including earlier runs."""
    entries = []
    for json_path in sorted(out_dir.glob("*.json")):
        txt_path = json_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not str(meta.get("tool", "")).startswith("transcribe"):
            continue
        rows = meta.get("segments", [])
        flagged = sum(1 for r in rows if r.get("low_confidence"))
        entries.append((meta, txt_path.name, len(rows), flagged))
    if not entries:
        return 0

    md = [
        "# Transcripts",
        "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Generated locally by transcribe {__version__}.",
        "",
        f"| Source file | Duration | Segments | {UNSURE} | Speakers | Model | Transcript |",
        "|---|---|---|---|---|---|---|",
    ]
    for meta, txt_name, count, flagged in entries:
        source = meta.get("source", {})
        md.append(
            f"| {md_cell(source.get('file', '?'))} | {clock(source.get('duration_sec', 0))} "
            f"| {count} | {flagged} | {'yes' if meta.get('diarization') else '-'} "
            f"| {md_cell(meta.get('model', '?'))} | `{txt_name}` |"
        )
    md += [
        "",
        f"**{UNSURE} = low confidence.** Listen to those parts before relying on a quote. "
        "Unmarked text can be wrong too: machine transcripts must be checked against the audio.",
        "",
    ]
    write_atomic(out_dir / "INDEX.md", "\n".join(md))
    return len(entries)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Private, local Hebrew speech-to-text with optional speaker labels.",
        epilog="Read the Disclaimer in README.md before use. "
               "Verify every transcript against the original audio.",
    )
    p.add_argument("inputs", nargs="*", metavar="PATH",
                   help="audio or video files and/or folders")
    p.add_argument("-o", "--out", default="transcripts",
                   help="output folder (default: ./transcripts)")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also look inside sub-folders")
    p.add_argument("-m", "--model", default="turbo",
                   help="'turbo' (default), 'full' (slower, sometimes more accurate), or any "
                        "faster-whisper model: a Hugging Face id or a local folder")
    p.add_argument("-l", "--language", default="he",
                   help="language code (default: he), or 'auto' to detect it")
    p.add_argument("--prompt", metavar="TEXT",
                   help="names and terms that occur in the audio, to help recognise them")
    p.add_argument("--diarize", action="store_true",
                   help="label speakers (needs a free Hugging Face token, see README)")
    p.add_argument("--speakers", type=int, default=0, metavar="N",
                   help="number of speakers, with --diarize (default: 0 = detect)")
    p.add_argument("--diarization-model", metavar="ID",
                   help="pyannote pipeline (default: community-1 on pyannote.audio 4.x)")
    p.add_argument("--flag-below", type=float, default=LOW_CONFIDENCE, metavar="X",
                   help=f"mark segments below this average log-probability "
                        f"(default: {LOW_CONFIDENCE}; higher marks more)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="default: auto (an NVIDIA GPU if available, else the CPU)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4, metavar="N",
                   help="CPU threads (default: all)")
    p.add_argument("-f", "--force", action="store_true",
                   help="transcribe again even if a transcript exists")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="do not print the text while transcribing")
    p.add_argument("--download-only", action="store_true",
                   help="download the model and exit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)
    if not args.inputs and not args.download_only:
        p.error("give at least one audio file or folder, e.g.:  python transcribe.py ~/Recordings")
    if args.speakers < 0:
        p.error("--speakers must be 0 (detect) or more")
    return args


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a console without UTF-8
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    args.model_name = MODELS.get(args.model, args.model)
    args.language = None if args.language.lower() == "auto" else args.language

    if args.download_only:
        resolve_model(args.model_name)
        print(f"Model ready: {args.model_name}")
        return 0

    files = find_inputs(args.inputs, args.recursive)
    if not files:
        print("Nothing to transcribe.")
        return 1

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(files)

    todo, failures = [], 0
    for src in files:
        name = names[src]
        outputs = {(out_dir / f"{name}{ext}").resolve() for ext in (".txt", ".srt", ".json")}
        if src.resolve() in outputs:  # never overwrite a source file
            print(f"x {src.name}: skipped, its transcript would overwrite it. Choose another --out.")
            failures += 1
        elif (out_dir / f"{name}.txt").exists() and not args.force:
            print(f"- {src.name}: already transcribed (--force to redo)")
        else:
            todo.append(src)

    if todo:
        model_dir = resolve_model(args.model_name)
        device = args.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        diarizer = load_diarizer(args, device) if args.diarize else None
        print(f"Loading {args.model_name} ...", flush=True)
        model = load_whisper(model_dir, device, args.threads)
        print(f"  ready ({device})\n")

        for i, src in enumerate(todo, 1):
            name = names[src]
            print(f"[{i}/{len(todo)}] {src.name}", flush=True)
            try:
                meta = transcribe_file(src, model, diarizer, args)
                write_outputs(meta, out_dir, name)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failures += 1
                print(f"   x failed: {type(e).__name__}: {e}\n")
                continue
            rows = meta["segments"]
            flagged = sum(1 for r in rows if r["low_confidence"])
            print(f"   done: {name}.txt  ({len(rows)} segments, {flagged} marked {UNSURE}, "
                  f"{meta['processing_sec'] / 60:.1f} min)\n")

    count = write_index(out_dir)
    print(f"Finished. {count} transcript(s) in {out_dir.resolve()}")
    if count:
        print(f"Check every transcript against the audio, starting with the {UNSURE} marks.")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        sys.exit(130)
