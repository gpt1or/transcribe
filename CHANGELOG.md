# Changelog

## 1.2.0 (2026-11-26)

- **Russian.** `--language ru` uses Ivan Bondarenko's Russian fine-tune of Whisper
  large-v3-turbo (`bond005/whisper-podlodka-turbo-ct2`); with `--model full`, OpenAI's
  multilingual large-v3. Other languages use OpenAI's multilingual models.
- **Language per file.** `--language auto` detects each file's language and uses the
  matching model; `--language he,ru` chooses only among the listed languages.
  Transcripts record `detected_language` and `language_probability`, and `INDEX.md`
  has a Language column.
- **Russian messages.** `--ui-language ru` (or `TRANSCRIBE_UI_LANGUAGE=ru`) puts
  messages, help, errors and the headers of `.txt`, `.srt` and `INDEX.md` in Russian,
  including `run.sh` and `setup.sh`. The `.json` data is the same in every language.
- **Disclaimer in Russian** in the README, plus a short Russian introduction.
- Python: `LANGUAGE_MODELS`, `DETECTION_MODEL`, `model_for()`, `required_models()`,
  `set_ui_language()`, `get_ui_language()`. `MODELS` still names the Hebrew models.
- `--download-only` downloads every model the chosen language needs; `setup.sh` takes
  `--language`.

## 1.1.0 (2026-11-10)

- **Installable as a Python package, `hebrew-transcribe`**:
  `pip install "hebrew-transcribe @ git+https://github.com/gpt1or/transcribe.git"`
  or `pipx install "git+https://github.com/gpt1or/transcribe.git"`.
  Commands `transcribe` and `hebrew-transcribe`, or `python -m hebrew_transcribe`.
- **Python API**: `Transcriber`, `Transcript`, `Segment`, `transcribe()`, `write_index()`,
  `resolve_model()`, `use_system_certificates()`; errors `ModelDownloadError` and
  `DiarizationUnavailable`. The library logs instead of printing and does not change
  process-wide settings beyond two telemetry defaults.
- Speaker labels are the optional extra `[diarize]`.
- The Hugging Face token is now saved where `hf auth login` saves it. A `.hf_token`
  file from 1.0 is still used by `run.sh`.
- `run.sh` and `setup.sh` install the package into `.venv` in editable mode;
  `requirements*.txt` and the top-level `transcribe.py` were replaced by `pyproject.toml`
  and `src/hebrew_transcribe/`.
- Offline test suite (the models are replaced by fakes) and GitHub workflows for tests
  and for publishing to PyPI with trusted publishing.

## 1.0.0 (2026-11-09)

- First release.
