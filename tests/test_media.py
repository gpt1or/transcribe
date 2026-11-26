"""Real audio and video files, decoded with the real faster-whisper and PyAV.

The clips in tests/media are one-second tones (no speech) in common formats:
m4a (AAC), mp4 and mov (H.264 + AAC), mkv (MPEG-4 + MP3), webm (VP9 + Opus),
and an mp4 without a sound track.

This runs in a separate Python process, because the other tests replace those
packages with fakes, and is skipped when they are not installed. No speech
model is needed: decoding, the voice detector and the parts of the
faster-whisper API this package calls are checked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

MEDIA = Path(__file__).resolve().parent / "media"

CHECK = r"""
import inspect, json, sys
import faster_whisper
from hebrew_transcribe import NoAudioError, load_audio
from hebrew_transcribe.core import _audio_tracks, _speech_chunks

report = {"files": {}}
for name in sys.argv[1:]:
    entry = {"tracks": _audio_tracks(name)}
    try:
        audio = load_audio(name)
        entry.update(seconds=len(audio) / 16000, dtype=str(audio.dtype),
                     chunks=len(_speech_chunks(audio)))
    except NoAudioError as e:
        entry["no_audio"] = str(e)
    report["files"][name] = entry

model = faster_whisper.WhisperModel
report["detect_language"] = sorted(inspect.signature(model.detect_language).parameters)
report["transcribe"] = sorted(inspect.signature(model.transcribe).parameters)
print(json.dumps(report))
"""


def _real_packages() -> bool:
    try:
        probe = subprocess.run([sys.executable, "-c", "import faster_whisper, av"],
                               capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@unittest.skipUnless(MEDIA.is_dir(), "tests/media is not here")
@unittest.skipUnless(_real_packages(), "faster-whisper and PyAV are not installed")
class RealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = sorted(str(p) for p in MEDIA.iterdir() if not p.name.startswith("."))
        env = dict(os.environ, TRANSCRIBE_UI_LANGUAGE="en")
        done = subprocess.run([sys.executable, "-c", CHECK, *files], capture_output=True,
                              text=True, timeout=600, env=env)
        if done.returncode:
            raise AssertionError(done.stderr[-4000:])
        report = json.loads(done.stdout.strip().splitlines()[-1])
        cls.files = {Path(name).name: entry for name, entry in report["files"].items()}
        cls.report = report

    def test_audio_and_video_files_decode(self):
        for name in ("tone.m4a", "tone.mp4", "tone.mov", "tone.mkv", "tone.webm"):
            with self.subTest(file=name):
                entry = self.files[name]
                self.assertEqual(entry["tracks"], 1)
                self.assertAlmostEqual(entry["seconds"], 1.0, delta=0.15)
                self.assertEqual(entry["dtype"], "float32")
                self.assertIsInstance(entry["chunks"], int)  # the voice detector ran

    def test_a_video_without_sound(self):
        entry = self.files["no_audio.mp4"]
        self.assertEqual(entry["tracks"], 0)
        self.assertEqual(entry["no_audio"], "no_audio.mp4 has no audio track")

    def test_the_faster_whisper_api_in_use(self):
        self.assertLessEqual({"audio", "vad_filter", "language_detection_segments",
                              "language_detection_threshold"}, set(self.report["detect_language"]))
        self.assertLessEqual({"audio", "language", "beam_size", "vad_filter", "vad_parameters",
                              "condition_on_previous_text", "initial_prompt", "temperature",
                              "no_speech_threshold", "log_prob_threshold"},
                             set(self.report["transcribe"]))


if __name__ == "__main__":
    unittest.main()
