"""Command line:  transcribe <files or folders> [options]   (also: hebrew-transcribe,
python -m hebrew_transcribe). Run with --help for all options."""
from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

from ._version import __version__
from .core import (
    LOW_CONFIDENCE,
    MODELS,
    UNSURE,
    DiarizationUnavailable,
    Diarizer,
    ModelDownloadError,
    Segment,
    Transcriber,
    clock,
    cuda_available,
    find_audio,
    find_hf_token,
    resolve_model,
    use_system_certificates,
)
from .output import output_names, save, write_index

TOKEN_HELP = """
Speaker labels use pyannote's models from Hugging Face. They are free but
gated, so a free access token is needed once:
  1. Create an account:        https://huggingface.co/join
  2. Accept the model terms:   https://huggingface.co/pyannote/speaker-diarization-community-1
  3. Create a "Read" token:    https://huggingface.co/settings/tokens
"""

NETWORK_HELP = """
  Internet is needed only for this one-time download. Try, in order:
    1. Disconnect from a work VPN, or switch network (a phone hotspot works),
       then run the same command again. The download resumes where it stopped.
    2. From a clone of the repository on macOS, behind a corporate proxy:
       bash setup.sh --corp-certs
"""


def _prepare_process() -> None:
    """Command-line only (the library never changes process-wide settings):
    trust the operating system's certificates, which is what makes downloads
    work behind TLS-inspecting corporate proxies, and skip Hugging Face's Xet
    download path, whose separate CDN some proxies block (HF_HUB_DISABLE_XET=0
    to re-enable it)."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    use_system_certificates()


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("hebrew_transcribe")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _save_token(token: str) -> Path:
    """Store the token where `hf auth login` would, readable only by the user."""
    try:
        from huggingface_hub import constants

        path = Path(constants.HF_TOKEN_PATH)
    except Exception:
        path = Path.home() / ".cache" / "huggingface" / "token"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def ask_token() -> str | None:
    """One-time interactive prompt; typing is hidden."""
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
        path = _save_token(token)
        print(f"  saved to {path} (readable only by you; remove with: hf auth logout)")
    except OSError as e:
        print(f"  ! could not save the token ({e}); using it for this run only")
    return token


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Private, local Hebrew speech-to-text with optional speaker labels.",
        epilog="Read the Disclaimer in the README before use. "
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
                   help="label speakers (needs the [diarize] extra and a free Hugging Face "
                        "token, see README)")
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
    p.add_argument("--version", action="version", version=f"hebrew-transcribe {__version__}")
    args = p.parse_args(argv)
    if not args.inputs and not args.download_only:
        p.error("give at least one audio file or folder, e.g.:  transcribe ~/Recordings")
    if args.speakers < 0:
        p.error("--speakers must be 0 (detect) or more")
    return args


def _download_failed(error: ModelDownloadError) -> int:
    print(f"x {error}")
    message = str(error)
    if "not found or is not accessible" not in message:
        print(NETWORK_HELP)
        if "403" in message or "Forbidden" in message:
            print("  (403 Forbidden usually means a proxy or firewall blocks the download.)")
    return 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a console without UTF-8
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    _setup_logging()
    _prepare_process()
    model_name = MODELS.get(args.model, args.model)

    if args.download_only:
        try:
            resolve_model(model_name)
        except ModelDownloadError as e:
            return _download_failed(e)
        print(f"Model ready: {model_name}")
        return 0

    files = find_audio(args.inputs, args.recursive)
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
        device = args.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"

        diarizer = None
        if args.diarize:
            token = find_hf_token() or ask_token()
            print("Loading the speaker-separation model...", flush=True)
            try:
                diarizer = Diarizer(token, args.diarization_model, device)
                print(f"  ready: {diarizer.model}\n")
            except DiarizationUnavailable as e:
                print(f"! {e}\n  Continuing without speaker labels.\n")

        print(f"Loading {model_name} ...", flush=True)
        try:
            transcriber = Transcriber(
                model_name,
                language=args.language,
                device=device,
                threads=args.threads,
                diarizer=diarizer,
                prompt=args.prompt,
                flag_below=args.flag_below,
            )
        except ModelDownloadError as e:
            return _download_failed(e)
        print(f"  ready ({transcriber.device})\n")

        def show(segment: Segment) -> None:
            print(f"      {clock(segment.start)}  {segment.text[:70]}", flush=True)

        for i, src in enumerate(todo, 1):
            name = names[src]
            print(f"[{i}/{len(todo)}] {src.name}", flush=True)
            try:
                result = transcriber.transcribe(
                    src, speakers=args.speakers or None, on_segment=None if args.quiet else show
                )
                save(result, out_dir, name)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failures += 1
                print(f"   x failed: {type(e).__name__}: {e}\n")
                continue
            print(f"   done: {name}.txt  ({len(result.segments)} segments, {result.flagged} marked "
                  f"{UNSURE}, {result.processing_sec / 60:.1f} min)\n")

    count = write_index(out_dir)
    print(f"Finished. {count} transcript(s) in {out_dir.resolve()}")
    if count:
        print(f"Check every transcript against the audio, starting with the {UNSURE} marks.")
    return 1 if failures else 0


def run() -> None:
    """Console-script entry point."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        sys.exit(130)
