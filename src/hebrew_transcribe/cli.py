"""Command line:  transcribe <files or folders> [options]   (also: hebrew-transcribe,
python -m hebrew_transcribe). Run with --help for all options; add
--ui-language ru for Russian messages."""
from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from pathlib import Path

from ._version import __version__
from .core import (
    DETECTION_MODEL,
    LOW_CONFIDENCE,
    SIZES,
    UNSURE,
    DiarizationUnavailable,
    Diarizer,
    ModelDownloadError,
    NoAudioError,
    Segment,
    Transcriber,
    clock,
    cuda_available,
    find_media,
    find_hf_token,
    model_for,
    parse_language,
    required_models,
    resolve_model,
    use_system_certificates,
)
from .i18n import UI_LANGUAGES, get_ui_language, set_ui_language, t
from .output import output_names, save, write_index


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
    print(t("token_help"))
    try:
        token = getpass.getpass(t("token_prompt")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not token:
        return None
    try:
        print(t("token_saved", path=_save_token(token)))
    except OSError as e:
        print(t("token_not_saved", error=e))
    return token


# ----------------------------------------------------------------------------
# Arguments, with help and errors in the UI language
# ----------------------------------------------------------------------------
class _Formatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        return super().add_usage(usage, actions, groups, t("cli.usage") if prefix is None else prefix)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, t("cli.error", prog=self.prog, message=message) + "\n")


def _ui_language_default() -> str:
    value = os.environ.get("TRANSCRIBE_UI_LANGUAGE", "").strip().lower()
    return value if value in UI_LANGUAGES else get_ui_language()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # First the UI language alone, so that the help and errors are in it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ui-language", choices=UI_LANGUAGES, default=_ui_language_default())
    known, _ = pre.parse_known_args(argv)
    set_ui_language(known.ui_language)

    p = _Parser(prog="transcribe", description=t("cli.description"), epilog=t("cli.epilog"),
                formatter_class=_Formatter, add_help=False)
    p._positionals.title = t("cli.positionals")
    p._optionals.title = t("cli.options")
    p.add_argument("-h", "--help", action="help", help=t("help.help"))
    p.add_argument("inputs", nargs="*", metavar="PATH", help=t("help.inputs"))
    p.add_argument("-o", "--out", default="transcripts", help=t("help.out"))
    p.add_argument("-r", "--recursive", action="store_true", help=t("help.recursive"))
    p.add_argument("-m", "--model", default="turbo", help=t("help.model"))
    p.add_argument("-l", "--language", default="he", help=t("help.language"))
    p.add_argument("--prompt", metavar="TEXT", help=t("help.prompt"))
    p.add_argument("--diarize", action="store_true", help=t("help.diarize"))
    p.add_argument("--speakers", type=int, default=0, metavar="N", help=t("help.speakers"))
    p.add_argument("--diarization-model", metavar="ID", help=t("help.diarization_model"))
    p.add_argument("--flag-below", type=float, default=LOW_CONFIDENCE, metavar="X",
                   help=t("help.flag_below", default=LOW_CONFIDENCE))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help=t("help.device"))
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4, metavar="N",
                   help=t("help.threads"))
    p.add_argument("-f", "--force", action="store_true", help=t("help.force"))
    p.add_argument("-q", "--quiet", action="store_true", help=t("help.quiet"))
    p.add_argument("--download-only", action="store_true", help=t("help.download_only"))
    p.add_argument("--ui-language", choices=UI_LANGUAGES, default=known.ui_language,
                   help=t("help.ui_language"))
    p.add_argument("--version", action="version", version=f"hebrew-transcribe {__version__}",
                   help=t("help.version"))
    args = p.parse_args(argv)
    if not args.inputs and not args.download_only:
        p.error(t("error.no_inputs"))
    if args.speakers < 0:
        p.error(t("error.speakers"))
    try:
        parse_language(args.language)
    except ValueError as e:
        p.error(str(e))
    return args


def _download_failed(error: ModelDownloadError) -> int:
    print(t("error_prefix", error=error))
    message = str(error)
    if "RepositoryNotFoundError" not in message and "GatedRepoError" not in message \
            and "RevisionNotFoundError" not in message:
        print(t("network_help"))
        if "403" in message or "Forbidden" in message:
            print(t("hint_403"))
    return 1


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a console without UTF-8
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    _setup_logging()
    _prepare_process()

    if args.download_only:
        for name in required_models(args.model, args.language):
            try:
                resolve_model(name)
            except ModelDownloadError as e:
                return _download_failed(e)
            print(t("model_ready", model=name))
        return 0

    files = find_media(args.inputs, args.recursive)
    if not files:
        print(t("nothing"))
        return 1

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(files)

    todo, failures = [], 0
    for src in files:
        name = names[src]
        outputs = {(out_dir / f"{name}{ext}").resolve() for ext in (".txt", ".srt", ".json")}
        if src.resolve() in outputs:  # never overwrite a source file
            print(t("would_overwrite", name=src.name))
            failures += 1
        elif (out_dir / f"{name}.txt").exists() and not args.force:
            print(t("already_done", name=src.name))
        else:
            todo.append(src)

    if todo:
        device = args.device
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"

        diarizer = None
        if args.diarize:
            token = find_hf_token() or ask_token()
            print(t("loading_diarizer"), flush=True)
            try:
                diarizer = Diarizer(token, args.diarization_model, device)
                print(t("diarizer_ready", model=diarizer.model) + "\n")
            except DiarizationUnavailable as e:
                print(f"! {e}\n{t('without_speakers')}\n")

        fixed, _ = parse_language(args.language)
        custom = None if args.model in SIZES else args.model
        if fixed is None:
            print(t("loading_detector", model=custom or DETECTION_MODEL), flush=True)
        else:
            print(t("loading_model", model=custom or model_for(fixed, args.model)), flush=True)
        try:
            transcriber = Transcriber(
                args.model,
                language=args.language,
                device=device,
                threads=args.threads,
                diarizer=diarizer,
                prompt=args.prompt,
                flag_below=args.flag_below,
            )
        except ModelDownloadError as e:
            return _download_failed(e)
        print(t("ready", device=transcriber.device) + "\n")

        def show(segment: Segment) -> None:
            print(f"      {clock(segment.start)}  {segment.text[:70]}", flush=True)

        for i, src in enumerate(todo, 1):
            name = names[src]
            print(t("file_header", index=i, total=len(todo), name=src.name), flush=True)
            try:
                result = transcriber.transcribe(
                    src, speakers=args.speakers or None, on_segment=None if args.quiet else show
                )
                save(result, out_dir, name)
            except KeyboardInterrupt:
                raise
            except NoAudioError as e:  # e.g. a video without sound
                failures += 1
                print(t("file_skipped", error=e) + "\n")
                continue
            except Exception as e:
                failures += 1
                print(t("file_failed", kind=type(e).__name__, error=e) + "\n")
                continue
            print(t("file_done", name=name, segments=len(result.segments), flagged=result.flagged,
                    mark=UNSURE, minutes=f"{result.processing_sec / 60:.1f}") + "\n")

    count = write_index(out_dir)
    print(t("finished", count=count, path=out_dir.resolve()))
    if count:
        print(t("check_marks", mark=UNSURE))
    return 1 if failures else 0


def run() -> None:
    """Console-script entry point."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n" + t("stopped"))
        sys.exit(130)
