"""Transcript files (.txt, .srt, .json) and the INDEX.md of an output folder.

The readable files (.txt, .srt, INDEX.md) use the current UI language (see
i18n); the .json data is always the same, whatever the UI language."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ._version import __version__
from .core import UNSURE, Transcript, clock, srt_time
from .i18n import speaker_name, t


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so an interrupted run never leaves a
    half-written transcript that a later run would mistake for a finished one."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _shares(tr: Transcript) -> str:
    """'he 70%, ru 30%' for a conversation in several languages."""
    ordered = sorted((tr.languages or {}).items(), key=lambda kv: -kv[1])
    return ", ".join(f"{code} {round(share * 100)}%" for code, share in ordered)


def language_text(tr: Transcript) -> str:
    """'he'; 'auto → ru (97%)' when the language was detected; or
    'auto → he 70%, ru 30% (mixed)' for a conversation in several languages."""
    detected = tr.detected_language
    if tr.mixed:
        return f"{tr.language} → {_shares(tr)} ({t('txt.mixed')})"
    if tr.language_probability is None or not detected or tr.language == detected:
        return tr.language or detected or "?"
    return f"{tr.language} → {detected} ({round(tr.language_probability * 100)}%)"


def model_text(tr: Transcript) -> str:
    """The model, or 'he: …; ru: …' when each language had its own."""
    if tr.models and len(tr.models) > 1:
        return "; ".join(f"{code}: {name}" for code, name in tr.models.items())
    return tr.model


def render_txt(tr: Transcript) -> str:
    line = "=" * 66
    fields = [
        (t("txt.source"), tr.file),
        (t("txt.sha256"), tr.sha256),
        (t("txt.duration"), clock(tr.duration)),
        (t("txt.model"), model_text(tr)),
        (t("txt.language"), language_text(tr)),
        (t("txt.speakers"), tr.diarization or t("txt.not_separated")),
        (t("txt.created"), t("txt.created_value", created=tr.created, tool=tr.tool)),
    ]
    width = max(len(label) for label, _ in fields)
    header = [t("txt.title", file=tr.file), line]
    header += [f"{label:<{width}} : {value}" for label, value in fields]
    header += ["-" * 66, t("txt.warning", mark=UNSURE), line]

    body: list[str] = []
    previous: object = object()
    for s in tr.segments:
        if tr.diarization and s.speaker != previous:
            if body:
                body.append("")
            body.append(f"{speaker_name(s.speaker)}:")
            previous = s.speaker
        mark = f" {UNSURE}" if s.low_confidence else ""
        tag = f"[{s.language}] " if tr.mixed and s.language else ""
        body.append(f"  [{clock(s.start)}] {tag}{s.text}{mark}")
    footer = [line, t("txt.end", segments=len(tr.segments), flagged=tr.flagged, mark=UNSURE), ""]
    return "\n".join(header + [""] + body + [""] + footer)


def render_srt(tr: Transcript) -> str:
    blocks = []
    for i, s in enumerate(tr.segments, 1):
        prefix = f"{speaker_name(s.speaker)}: " if s.speaker else ""
        blocks.append(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{prefix}{s.text}\n")
    return "\n".join(blocks)


def save(tr: Transcript, out_dir: str | os.PathLike, name: str | None = None) -> Path:
    """Write <name>.json, <name>.srt and <name>.txt into out_dir. The .txt is
    written last: its presence is what marks a file as done. Returns its path."""
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    name = name or Path(tr.file).stem
    write_atomic(folder / f"{name}.json", tr.to_json())
    write_atomic(folder / f"{name}.srt", render_srt(tr))
    txt = folder / f"{name}.txt"
    write_atomic(txt, render_txt(tr))
    return txt


def output_names(files: Iterable[Path]) -> dict[Path, str]:
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


def _md_cell(text: object) -> str:
    return str(text).replace("|", "\\|")


def _index_language(tr: Transcript) -> str:
    if tr.mixed:
        return _shares(tr)
    code = tr.detected_language or tr.language or "?"
    if tr.language_probability is None:
        return code
    return f"{code} ({round(tr.language_probability * 100)}%)"


def write_index(out_dir: str | os.PathLike) -> int:
    """(Re)write INDEX.md for every transcript in the folder, including those
    from earlier runs. Returns how many transcripts it lists."""
    folder = Path(out_dir).expanduser()
    entries = []
    for json_path in sorted(folder.glob("*.json")):
        txt_path = json_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Files written by this tool (version 1.0 called itself "transcribe").
        if not str(data.get("tool", "")).startswith(("hebrew-transcribe", "transcribe")):
            continue
        entries.append((Transcript.from_dict(data), txt_path.name))
    if not entries:
        return 0

    columns = t("index.columns", mark=UNSURE)
    md = [
        t("index.title"),
        "",
        t("index.updated", date=f"{datetime.now():%Y-%m-%d %H:%M}", version=__version__),
        "",
        f"| {columns} |",
        "|" + "---|" * (columns.count("|") + 1),
    ]
    for tr, txt_name in entries:
        md.append(
            f"| {_md_cell(tr.file or '?')} | {clock(tr.duration)} | {len(tr.segments)} | {tr.flagged} "
            f"| {t('index.yes') if tr.diarization else '-'} | {_md_cell(_index_language(tr))} "
            f"| {_md_cell(model_text(tr) or '?')} | `{txt_name}` |"
        )
    md += ["", t("index.note", mark=UNSURE), ""]
    write_atomic(folder / "INDEX.md", "\n".join(md))
    return len(entries)
