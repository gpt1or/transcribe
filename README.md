# transcribe

**Private, local Hebrew speech-to-text, with optional speaker labels.** Python package: `hebrew-transcribe`.

Turn recordings (phone calls, meetings, voice notes, videos) into timestamped text on your own computer, from the command line or from Python. After a one-time model download it works offline, and it never sends your audio anywhere.

> [!IMPORTANT]
> Read the **[Disclaimer](#disclaimer)** before using this software ([Hebrew version below](#disclaimer-he)). You alone are responsible for having the legal right to process each recording, and for checking every transcript against the original audio.

## Features

- **Tuned for Hebrew.** Uses [ivrit.ai](https://www.ivrit.ai)'s Hebrew fine-tune of OpenAI's Whisper large-v3-turbo, run locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Other languages work with `--language` and a suitable `--model`.
- **Three files per recording.** `.txt` (readable, with timestamps), `.srt` (subtitles, for checking the text against the audio in a player) and `.json` (segments with confidence scores), plus an `INDEX.md` of everything transcribed so far.
- **Doubtful passages are marked.** Low-confidence segments get `⟨?⟩`, so a human checks them first.
- **Optional speaker labels.** `--diarize` separates voices into "Speaker 1", "Speaker 2" with [pyannote](https://github.com/pyannote/pyannote-audio). Off by default.
- **Source files are never touched.** They are opened read-only, never converted or moved. Each transcript records the SHA-256 of its source file, so it can be matched to the exact file it came from.
- **Resumable.** Stop with Ctrl-C at any time; run the same command again and it continues.
- **A command and a library.** `transcribe` on the command line, `from hebrew_transcribe import Transcriber` in Python.
- **Telemetry off.** The usage metrics of the underlying libraries are disabled.

## Install

Needs Python 3.10 or newer and about 3 GB of free disk space. The speech model (about 1.6 GB) is downloaded the first time you transcribe.

**As a command-line tool**, with [pipx](https://pipx.pypa.io), which keeps it in its own environment:

```bash
pipx install "git+https://github.com/gpt1or/transcribe.git"
transcribe ~/path/to/recordings
```

**Into a Python environment**, to use it from your own code (or without pipx):

```bash
pip install "hebrew-transcribe @ git+https://github.com/gpt1or/transcribe.git"
```

**As a dependency of another project**, pinned to a release:

```text
# requirements.txt
hebrew-transcribe @ git+https://github.com/gpt1or/transcribe.git@v1.1.0
```

```toml
# pyproject.toml
dependencies = ["hebrew-transcribe @ git+https://github.com/gpt1or/transcribe.git@v1.1.0"]
```

**From a clone, without installing anything system-wide** (macOS / Linux):

```bash
git clone https://github.com/gpt1or/transcribe.git
cd transcribe
bash run.sh ~/path/to/recordings
```

`run.sh` installs the package into a private `.venv` folder on the first run (a few minutes, once) and then works exactly like the `transcribe` command.

The program is available as `transcribe`, as `hebrew-transcribe` (the same program, in case another tool already uses the name `transcribe`) and as `python -m hebrew_transcribe`. Transcripts are written to `./transcripts/`. On Windows, use the pip or pipx commands above (Git must be installed), then run `transcribe C:\path\to\recordings`.

## Command line

```bash
transcribe call.m4a                               # one file
transcribe ~/Recordings --out ~/Transcripts       # a whole folder, custom output folder
transcribe ~/Recordings --diarize --speakers 2    # phone calls, with speaker labels
transcribe lecture.mp4 --model full               # slower model, sometimes more accurate
transcribe ~/Recordings --prompt "רונית, יוסי, משכנתא" # names and terms that occur in the audio
```

An invented example of the output:

```text
Speaker 1:
  [00:03] שלום, רציתי לקבוע תור לשבוע הבא

Speaker 2:
  [00:07] בטח, איזה יום נוח לך? ⟨?⟩
```

| Option | Meaning |
|---|---|
| `PATH ...` | audio or video files and/or folders: m4a, mp3, wav, aac, flac, ogg, opus, amr, mp4, mov, mkv and more |
| `-o`, `--out DIR` | output folder (default `./transcripts`) |
| `-r`, `--recursive` | also look inside sub-folders |
| `-m`, `--model` | `turbo` (default), `full` (about 3× slower, sometimes more accurate), or any faster-whisper model: a Hugging Face id or a local folder |
| `-l`, `--language` | language code (default `he`), or `auto` to detect it |
| `--prompt TEXT` | names and terms that occur in the audio, to help recognise them |
| `--diarize` | label speakers (see below) |
| `--speakers N` | number of speakers, if known (default: detect) |
| `--flag-below X` | `⟨?⟩` threshold, average log-probability (default `-0.55`; higher marks more) |
| `--device` | `auto` (default), `cpu` or `cuda` |
| `-f`, `--force` | transcribe again files that already have a transcript |
| `-q`, `--quiet` | do not print the text while working |
| `--download-only` | download the model and exit |

## Python

```python
from hebrew_transcribe import Transcriber, write_index

transcriber = Transcriber()                   # loads the model (downloads it the first time)
result = transcriber.transcribe("call.m4a")

print(result.text)                            # the whole text
for s in result.segments:                     # timestamps, confidence and speaker per segment
    print(f"{s.start:8.1f}  {s.text}{'  ⟨?⟩' if s.low_confidence else ''}")

result.save("transcripts")                    # call.txt, call.srt and call.json
write_index("transcripts")                    # INDEX.md of the folder
```

With speaker labels (needs the `diarize` extra and a Hugging Face token, see below):

```python
transcriber = Transcriber(diarize=True)       # token from HF_TOKEN, `hf auth login`, or hf_token="..."
result = transcriber.transcribe("call.m4a", speakers=2)
```

- `Transcriber(model="turbo", language="he", device="auto", prompt=None, flag_below=-0.55, diarize=False, ...)` loads the models once; reuse it for many files. `model` also accepts `"full"`, a Hugging Face id or a local folder; `language="auto"` detects the language.
- `transcribe(path, ..., speakers=None, prompt=None, on_segment=None)` returns a `Transcript` (`.segments`, `.text`, `.sha256`, `.duration`, ...). `on_segment` is called with each segment as soon as it is ready.
- `hebrew_transcribe.transcribe("call.m4a")` is a one-off shortcut; `Transcript.load("call.json")` reads a saved transcript back.
- Errors are raised as `ModelDownloadError` and `DiarizationUnavailable`. Progress goes to the `hebrew_transcribe` logger; the library never prints.
- The library does not change process-wide settings, apart from the two telemetry defaults described under [Privacy](#privacy). Behind a TLS-inspecting corporate proxy, call `hebrew_transcribe.use_system_certificates()` once at startup.

## Speaker labels (optional)

Speaker labels need extra packages (pyannote.audio and PyTorch, about 1 to 2 GB):

```bash
pipx inject hebrew-transcribe "pyannote.audio>=4.0"                                       # pipx
pip install "hebrew-transcribe[diarize] @ git+https://github.com/gpt1or/transcribe.git"   # pip
```

From a clone, `bash run.sh ... --diarize` installs them automatically.

The speaker model is free but gated on Hugging Face, so a free access token is needed once:

1. Create an account: <https://huggingface.co/join>
2. Accept the terms of <https://huggingface.co/pyannote/speaker-diarization-community-1> (with pyannote.audio 3.x: [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0))
3. Create a token with **Read** access: <https://huggingface.co/settings/tokens>

Then add `--diarize`. The first time, the command asks for the token; typing is hidden. It is saved where `hf auth login` saves it (`~/.cache/huggingface/token`), readable only by your user account; `hf auth logout` removes it. You can also set the `HF_TOKEN` environment variable instead.

Speaker labels separate voices; they do not identify anyone. Matching "Speaker 1" to a real person is up to you, and labels can be wrong, especially when people talk over each other.

## Why these settings

- **`condition_on_previous_text=False`.** Whisper tends to get stuck in loops and invent text during silence and noise, which phone calls are full of. This switches off the mechanism behind it.
- **Voice-activity detection** skips silence, for the same reason.
- **The `⟨?⟩` threshold of −0.55.** At −0.85 nothing was marked in about 90 minutes of phone audio; −0.55 marks what is worth a second listen without marking everything. Adjust with `--flag-below`.
- **`--prompt`.** Telling Whisper which names and terms to expect noticeably improves how they are written.
- **Read-only sources and SHA-256.** A transcript can always be tied to an unmodified original.

## Privacy

- Audio, transcripts and your token stay on your computer. This project has no server, and the author receives nothing.
- The network is used only to install packages (from PyPI) and to download models (from Hugging Face). Once downloaded, the speech model is loaded from disk without contacting the network; the optional speaker model may still check Hugging Face for updates (metadata only). To block all network access, set `HF_HUB_OFFLINE=1`.
- Telemetry is off. Importing the package sets `PYANNOTE_METRICS_ENABLED=0` (pyannote.audio's usage metrics) and `HF_HUB_DISABLE_TELEMETRY=1`, unless you have set them yourself.
- The command-line tool also trusts the operating system's certificate store and skips Hugging Face's Xet download CDN (`HF_HUB_DISABLE_XET=1`), which helps behind corporate proxies. The library leaves those choices to you.
- In a clone, `transcripts/`, common audio formats, tokens and certificate files are listed in `.gitignore`, so they are not committed by accident.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` | A corporate proxy is inspecting TLS. From a clone on macOS, run `bash setup.sh --corp-certs`; otherwise point `SSL_CERT_FILE` at your organisation's certificate bundle, or do the one-time download from another network. |
| `403 Forbidden` from a `cdn.hf.co` address | The network blocks Hugging Face's Xet CDN. Set `HF_HUB_DISABLE_XET=1` (the command-line tool does this by default). |
| The download keeps failing | Disconnect from a work VPN or switch network (a phone hotspot works) and run again. It resumes, and internet is needed only once. |
| `python3: command not found` | macOS: `xcode-select --install` or `brew install python`. Linux: install `python3` and `python3-venv`. |
| Speaker labels fail to load | Install the extra packages and accept the model terms (see above) with the account that owns the token. Transcription works without them. |
| Text repeats in a loop | Run that file again with `--model full --force`. |
| Disk space | Models are stored in `~/.cache/huggingface`: about 1.6 GB for `turbo`, about 3 GB for `full`. |
| It is slow | faster-whisper runs on the CPU (also on Apple Silicon) unless an NVIDIA GPU is available. `turbo`, the default, is the fast model; close heavy applications. |

## Development

```bash
git clone https://github.com/gpt1or/transcribe.git && cd transcribe
python3 -m venv .venv && .venv/bin/pip install -e ".[diarize]"
.venv/bin/python -m unittest discover -s tests -v    # offline: the models are replaced by fakes
```

To release: update `src/hebrew_transcribe/_version.py` and `CHANGELOG.md`, push, then publish a GitHub release tagged `vX.Y.Z`. The `publish` workflow uploads it to PyPI with trusted publishing (see the comment at the top of `.github/workflows/publish.yml` for the one-time setup).

## Third-party components

| Component | Used for | License |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | transcription engine | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | model inference | MIT |
| [ivrit-ai/whisper-large-v3-turbo-ct2](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) | default speech model | Apache-2.0 |
| [ivrit-ai/whisper-large-v3-ct2](https://huggingface.co/ivrit-ai/whisper-large-v3-ct2) | `--model full` | Apache-2.0 |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | speaker labels (optional) | MIT |
| [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | speaker model (optional, gated) | CC-BY-4.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | model downloads | Apache-2.0 |
| [truststore](https://github.com/sethmlarson/truststore) | system certificate store | MIT |

None of these is included in this repository. You download them yourself, from their official sources, when you install or run the tool, and each is governed by its own license and terms. The licenses above are as published by each project at the time of writing; check the source for current terms.

## Disclaimer

**Please read this before using the software. If you do not agree with it, do not use the software.**

This is a free, non-commercial, open-source project. It is provided **"AS IS", without warranty of any kind**, under the [MIT License](LICENSE). The terms below are in addition to that license.

1. **Your recordings are your responsibility.** This software does not record anything; it only processes audio files that you choose to give it. You alone are responsible for making sure that you are legally allowed to record, keep, transcribe, store and share every recording and transcript, including obtaining any consent required from the people recorded. Laws differ between countries and states: some allow a participant to record a conversation they take part in, others require the consent of everyone in it, and publishing a recording or its transcript may be restricted even where making it was lawful.
2. **Personal data and voice data.** Recordings and transcripts usually contain other people's personal data. The optional speaker-labelling feature (`--diarize`, off by default) analyses voice characteristics to tell speakers apart, and some laws specifically regulate voice or biometric data (for example, the EU General Data Protection Regulation and the Illinois Biometric Information Privacy Act). When you process other people's data, you are responsible for complying with the privacy and data-protection laws that apply to you.
3. **No guarantee of accuracy.** Machine transcription makes mistakes: wrong, missing or invented words, wrong timestamps and wrong speaker labels. The `⟨?⟩` marks catch only some of them, and unmarked text can be wrong too. Check every transcript against the original audio before relying on it, especially in legal, medical, financial, employment or other important matters. The output is not a certified transcription and is not, by itself, reliable evidence of what was said.
4. **Not legal advice.** Nothing in this repository is legal or other professional advice. If in doubt, consult a lawyer where you live.
5. **Third-party models and software.** The speech and speaker models are downloaded by you from Hugging Face, and the Python packages from PyPI. They are not part of this repository, are not distributed by the author, and are governed by their own licenses and terms, which you must accept and follow (see [Third-party components](#third-party-components)).
6. **No data collection.** The software runs on your own computer. The author operates no server and never receives, stores or has access to your audio, transcripts or access tokens.
7. **No affiliation.** This project is independent. It is not affiliated with, endorsed or sponsored by OpenAI, ivrit.ai, pyannote, Hugging Face or any other organisation; their names are used only to identify their components.
8. **No misuse.** Do not use this software for unlawful interception or eavesdropping, covert surveillance, stalking or harassment, or to process recordings obtained unlawfully.
9. **No support obligation.** The author has no obligation to provide support, maintenance, updates or fixes.
10. **Limitation of liability.** To the fullest extent permitted by applicable law, the author and contributors are not liable for any direct, indirect, incidental, special, consequential or punitive damages, or for any loss of data, profits or reputation, or for any claims, arising out of or in connection with the use of, misuse of, or inability to use this software or its output, including claims by third parties whose voices or data were processed. You use the software entirely at your own risk, and you agree to indemnify and hold harmless the author and contributors against any claim arising from your use of the software in breach of the law or of this notice.

<a name="disclaimer-he"></a>

<div dir="rtl">

## הצהרת אחריות והגבלת אחריות

**יש לקרוא לפני השימוש בתוכנה. מי שאינו מסכים לאמור כאן אינו רשאי להשתמש בה.**

זהו פרויקט קוד פתוח חינמי שאינו מסחרי. התוכנה מסופקת **"כמות שהיא" (AS IS), ללא אחריות מכל סוג**, לפי [רישיון MIT](LICENSE). התנאים שלהלן באים בנוסף לרישיון.

1. **ההקלטות באחריותך.** התוכנה אינה מקליטה דבר; היא רק מעבדת קובצי שמע שבחרת לתת לה. האחריות לוודא שמותר לך על פי דין להקליט, להחזיק, לתמלל, לשמור ולשתף כל הקלטה וכל תמלול, לרבות קבלת כל הסכמה הנדרשת מהמוקלטים, חלה עליך בלבד. הדין שונה ממדינה למדינה: יש המתירים למשתתף בשיחה להקליט אותה, ויש הדורשים את הסכמת כל המשתתפים. גם כשההקלטה עצמה מותרת, פרסומה או פרסום התמלול עשויים להיות אסורים.
2. **מידע אישי ומידע קולי.** הקלטות ותמלולים מכילים בדרך כלל מידע אישי על אחרים. הפרדת הדוברים (`--diarize`, כבויה כברירת מחדל) מנתחת מאפייני קול כדי להבחין בין דוברים, ויש דינים המסדירים במיוחד מידע קולי או ביומטרי. כשאתה מעבד מידע על אחרים, האחריות לעמוד בדיני הפרטיות והגנת המידע החלים עליך היא שלך.
3. **אין התחייבות לדיוק.** תמלול ממוכן טועה: מילים שגויות, חסרות או מומצאות, חותמות זמן שגויות ושיוך דוברים שגוי. הסימון `⟨?⟩` תופס רק חלק מהטעויות, וגם טקסט שאינו מסומן עלול להיות שגוי. יש לבדוק כל תמלול מול ההקלטה המקורית לפני שמסתמכים עליו, בפרט בעניינים משפטיים, רפואיים, כספיים או תעסוקתיים. הפלט אינו תמלול מאושר, ואינו כשלעצמו ראיה מהימנה למה שנאמר.
4. **אין זה ייעוץ משפטי.** דבר במאגר זה אינו ייעוץ משפטי או מקצועי אחר. במקרה של ספק יש להיוועץ בעורך דין.
5. **מודלים ותוכנות של צד שלישי.** מודלי הדיבור והדוברים מורדים על ידך מ-Hugging Face, וחבילות ה-Python מ-PyPI. הם אינם חלק ממאגר זה, אינם מופצים על ידי המפתח, וחלים עליהם הרישיונות והתנאים שלהם, שעליך לקבל ולקיים.
6. **אין איסוף מידע.** התוכנה פועלת על המחשב שלך. למפתח אין שרת, והוא אינו מקבל, אינו שומר ואין לו גישה להקלטות, לתמלולים או לטוקנים שלך.
7. **אין זיקה לגופים אחרים.** הפרויקט עצמאי, ואינו קשור ל-OpenAI, ל-ivrit.ai, ל-pyannote, ל-Hugging Face או לכל גוף אחר, ואינו נתמך או ממומן על ידם. שמותיהם משמשים רק לזיהוי הרכיבים.
8. **אין להשתמש לרעה.** אין להשתמש בתוכנה להאזנת סתר או ליירוט שלא כדין, למעקב סמוי, לבילוש או להטרדה, או לעיבוד הקלטות שהושגו שלא כדין.
9. **אין חובת תמיכה.** המפתח אינו מחויב לספק תמיכה, תחזוקה, עדכונים או תיקונים.
10. **הגבלת אחריות.** במידה המרבית המותרת על פי דין, המפתח והתורמים לא יישאו באחריות לכל נזק ישיר, עקיף, מקרי, מיוחד, תוצאתי או עונשי, או לאובדן מידע, רווחים או מוניטין, או לכל תביעה, הנובעים מהשימוש בתוכנה או בפלט שלה, משימוש לרעה בהם או מאי-יכולת להשתמש בהם, לרבות תביעות של צדדים שלישיים שקולם או המידע עליהם עובדו. השימוש בתוכנה הוא על אחריותך בלבד, ואתה מסכים לשפות את המפתח והתורמים ולפטור אותם מכל תביעה הנובעת משימושך בתוכנה בניגוד לדין או להודעה זו.

הנוסח העברי הוא תרגום לנוחות. בכל מקרה של סתירה, יגבר הנוסח האנגלי.

</div>

## License

[MIT](LICENSE) © 2026 gpt1or
