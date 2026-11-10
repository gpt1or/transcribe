# Changelog

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
