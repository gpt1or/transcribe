"""Transcript files (.txt, .srt, .json) and the INDEX.md of an output folder."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ._version import __version__
from .core import UNSURE, Transcript, clock, srt_time


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so an interrupted run never leaves a
    half-written transcript that a later run would mistake for a finished one."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def render_txt(t: Transcript) -> str:
    line = "=" * 66
    header = [
        f"Transcript - {t.file}",
        line,
        f"Source file : {t.file}",
        f"SHA-256     : {t.sha256}",
        f"Duration    : {clock(t.duration)}",
        f"Model       : {t.model}",
        f"Language    : {t.language}",
        f"Speakers    : {t.diarization or 'not separated'}",
        f"Created     : {t.created}, locally, by {t.tool}",
        "-" * 66,
        "Machine-generated. It may contain wrong, missing or invented words",
        f"and wrong speaker labels. {UNSURE} = low confidence. Verify against the",
        "original audio before relying on any part of it.",
        line,
    ]
    body: list[str] = []
    previous: object = object()
    for s in t.segments:
        if t.diarization and s.speaker != previous:
            if body:
                body.append("")
            body.append(f"{s.speaker or 'Speaker ?'}:")
            previous = s.speaker
        mark = f" {UNSURE}" if s.low_confidence else ""
        body.append(f"  [{clock(s.start)}] {s.text}{mark}")
    footer = [line, f"End of transcript. {len(t.segments)} segments, {t.flagged} marked {UNSURE}.", ""]
    return "\n".join(header + [""] + body + [""] + footer)


def render_srt(t: Transcript) -> str:
    blocks = []
    for i, s in enumerate(t.segments, 1):
        prefix = f"{s.speaker}: " if s.speaker else ""
        blocks.append(f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{prefix}{s.text}\n")
    return "\n".join(blocks)


def save(t: Transcript, out_dir: str | os.PathLike, name: str | None = None) -> Path:
    """Write <name>.json, <name>.srt and <name>.txt into out_dir. The .txt is
    written last: its presence is what marks a file as done. Returns its path."""
    folder = Path(out_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    name = name or Path(t.file).stem
    write_atomic(folder / f"{name}.json", t.to_json())
    write_atomic(folder / f"{name}.srt", render_srt(t))
    txt = folder / f"{name}.txt"
    write_atomic(txt, render_txt(t))
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

    md = [
        "# Transcripts",
        "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Generated locally by hebrew-transcribe {__version__}.",
        "",
        f"| Source file | Duration | Segments | {UNSURE} | Speakers | Model | Transcript |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, txt_name in entries:
        md.append(
            f"| {_md_cell(t.file or '?')} | {clock(t.duration)} | {len(t.segments)} | {t.flagged} "
            f"| {'yes' if t.diarization else '-'} | {_md_cell(t.model or '?')} | `{txt_name}` |"
        )
    md += [
        "",
        f"**{UNSURE} = low confidence.** Listen to those parts before relying on a quote. "
        "Unmarked text can be wrong too: machine transcripts must be checked against the audio.",
        "",
    ]
    write_atomic(folder / "INDEX.md", "\n".join(md))
    return len(entries)
