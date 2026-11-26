"""Messages of the command-line tool and the text of transcript files, in
English and Russian.

The language is process-wide. Choose it with set_ui_language("ru"), the
--ui-language option, or the TRANSCRIBE_UI_LANGUAGE environment variable.
It affects messages and the readable files (.txt, .srt, INDEX.md); the .json
data and the transcript text itself are never translated.
"""
from __future__ import annotations

import os
import re

UI_LANGUAGES = ("en", "ru")

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # finding inputs
        "no_audio": "  ! no audio or video files in {path}{hint}",
        "no_audio_hint": " (add --recursive for sub-folders)",
        "not_found": "  x not found: {path}",
        "no_audio_track": "{file} has no audio track",
        "empty_audio": "{file}: the audio track is empty",
        # models
        "downloading": "Downloading {model} (one time; resumes if interrupted)...",
        "attempt_failed": "  attempt {attempt}/{attempts} failed: {error}",
        "model_not_found": "Model '{model}' was not found or is not accessible ({kind}).",
        "download_failed": "Could not download '{model}': {error}",
        "loading_model": "Loading {model} ...",
        "loading_detector": "Loading the language-detection model {model} ...",
        "ready": "  ready ({device})",
        "model_ready": "Model ready: {model}",
        # language detection
        "detected": "   language: {language} ({percent}%) → {model}",
        "uncertain": "   ! the language is uncertain; check the result, or set it with --language",
        "no_speech_detected": "   ! no speech found for language detection; assuming {language}",
        "mixed": "   languages: {languages} (mixed conversation, {parts} parts)",
        "checking_parts": "   more than one language heard; checking every part of the recording...",
        "scan_failed": "   ! could not split the audio for language detection ({kind}); "
                       "using one language for the whole file",
        # speaker labels
        "separating": "   separating speakers...",
        "separation_failed": "   ! speaker separation failed ({kind}); continuing without it",
        "diarize_extra": "Speaker labels need extra packages: pip install 'hebrew-transcribe[diarize]' "
                         "(from a clone of the repository: bash setup.sh --diarize)",
        "diarize_cloud": "{model} is a cloud pipeline; only local pipelines are supported",
        "diarize_failed": "Could not load {model} ({kind}: {error}). Accept the model's terms on "
                          "Hugging Face with the account that owns the token, and check the token.",
        "diarize_no_access": "No access to {model}. Accept the model's terms on Hugging Face with "
                             "the account that owns the token, and check the token.",
        "loading_diarizer": "Loading the speaker-separation model...",
        "diarizer_ready": "  ready: {model}",
        "without_speakers": "  Continuing without speaker labels.",
        # Hugging Face token
        "token_help": (
            "\nSpeaker labels use pyannote's models from Hugging Face. They are free but\n"
            "gated, so a free access token is needed once:\n"
            "  1. Create an account:        https://huggingface.co/join\n"
            "  2. Accept the model terms:   https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  3. Create a \"Read\" token:    https://huggingface.co/settings/tokens\n"
        ),
        "token_prompt": "Hugging Face token (hidden; Enter to skip): ",
        "token_saved": "  saved to {path} (readable only by you; remove with: hf auth logout)",
        "token_not_saved": "  ! could not save the token ({error}); using it for this run only",
        # network
        "network_help": (
            "\n  Internet is needed only for this one-time download. Try, in order:\n"
            "    1. Disconnect from a work VPN, or switch network (a phone hotspot works),\n"
            "       then run the same command again. The download resumes where it stopped.\n"
            "    2. From a clone of the repository on macOS, behind a corporate proxy:\n"
            "       bash setup.sh --corp-certs\n"
        ),
        "hint_403": "  (403 Forbidden usually means a proxy or firewall blocks the download.)",
        # a run
        "nothing": "Nothing to transcribe.",
        "would_overwrite": "x {name}: skipped, its transcript would overwrite it. Choose another --out.",
        "already_done": "- {name}: already transcribed (--force to redo)",
        "file_header": "[{index}/{total}] {name}",
        "file_failed": "   x failed: {kind}: {error}",
        "file_skipped": "   x {error}",
        "file_done": "   done: {name}.txt  ({segments} segments, {flagged} marked {mark}, {minutes} min)",
        "finished": "Finished. {count} transcript(s) in {path}",
        "check_marks": "Check every transcript against the audio, starting with the {mark} marks.",
        "stopped": "Stopped. Run the same command again to continue where it left off.",
        "error_prefix": "x {error}",
        # command-line help
        "cli.description": "Private, local Hebrew and Russian speech-to-text for audio and video, "
                           "with optional speaker labels.",
        "cli.epilog": "Read the Disclaimer in the README before use. "
                      "Verify every transcript against the original audio.",
        "cli.usage": "usage: ",
        "cli.positionals": "positional arguments",
        "cli.options": "options",
        "cli.error": "{prog}: error: {message}",
        "help.inputs": "audio or video files and/or folders",
        "help.out": "output folder (default: ./transcripts)",
        "help.recursive": "also look inside sub-folders",
        "help.model": "'turbo' (default), 'full' (larger and slower), or any faster-whisper model: "
                      "a Hugging Face id or a local folder",
        "help.language": "language of the recordings: he (default), ru, any other language code, "
                         "'auto' to detect it (also when a conversation mixes languages), or a list "
                         "such as 'he,ru' to choose among those",
        "help.prompt": "names and terms that occur in the audio, to help recognise them",
        "help.diarize": "label speakers (needs the [diarize] extra and a free Hugging Face token, "
                        "see README)",
        "help.speakers": "number of speakers, with --diarize (default: 0 = detect)",
        "help.diarization_model": "pyannote pipeline (default: community-1 on pyannote.audio 4.x)",
        "help.flag_below": "mark segments below this average log-probability "
                           "(default: {default}; higher marks more)",
        "help.device": "default: auto (an NVIDIA GPU if available, else the CPU)",
        "help.threads": "CPU threads (default: all)",
        "help.force": "transcribe again even if a transcript exists",
        "help.quiet": "do not print the text while transcribing",
        "help.download_only": "download the model(s) and exit",
        "help.ui_language": "language of these messages and of the transcript headers: en or ru "
                            "(or set TRANSCRIBE_UI_LANGUAGE)",
        "help.help": "show this help message and exit",
        "help.version": "show the version and exit",
        "error.no_inputs": "give at least one audio file or folder, e.g.:  transcribe ~/Recordings",
        "error.speakers": "--speakers must be 0 (detect) or more",
        "error.language": "unknown language code: {code}",
        # transcript files
        "txt.title": "Transcript - {file}",
        "txt.source": "Source file",
        "txt.sha256": "SHA-256",
        "txt.duration": "Duration",
        "txt.model": "Model",
        "txt.language": "Language",
        "txt.speakers": "Speakers",
        "txt.created": "Created",
        "txt.created_value": "{created}, locally, by {tool}",
        "txt.not_separated": "not separated",
        "txt.mixed": "mixed",
        "txt.warning": "Machine-generated. It may contain wrong, missing or invented words\n"
                       "and wrong speaker labels. {mark} = low confidence. Verify against the\n"
                       "original audio before relying on any part of it.",
        "txt.end": "End of transcript. {segments} segments, {flagged} marked {mark}.",
        "speaker": "Speaker {n}",
        "speaker_unknown": "Speaker ?",
        "index.title": "# Transcripts",
        "index.updated": "Updated {date}. Generated locally by hebrew-transcribe {version}.",
        "index.columns": "Source file | Duration | Segments | {mark} | Speakers | Language | Model | Transcript",
        "index.yes": "yes",
        "index.note": "**{mark} = low confidence.** Listen to those parts before relying on a quote. "
                      "Unmarked text can be wrong too: machine transcripts must be checked against the audio.",
    },
    "ru": {
        # поиск файлов
        "no_audio": "  ! в {path} нет аудио- или видеофайлов{hint}",
        "no_audio_hint": " (для вложенных папок добавьте --recursive)",
        "not_found": "  x не найдено: {path}",
        "no_audio_track": "в файле {file} нет звуковой дорожки",
        "empty_audio": "{file}: звуковая дорожка пуста",
        # модели
        "downloading": "Скачивание {model} (однократно; при обрыве продолжится с того же места)...",
        "attempt_failed": "  попытка {attempt}/{attempts} не удалась: {error}",
        "model_not_found": "Модель «{model}» не найдена или недоступна ({kind}).",
        "download_failed": "Не удалось скачать «{model}»: {error}",
        "loading_model": "Загрузка модели {model} ...",
        "loading_detector": "Загрузка модели определения языка {model} ...",
        "ready": "  готово ({device})",
        "model_ready": "Модель готова: {model}",
        # определение языка
        "detected": "   язык: {language} ({percent}%) → {model}",
        "uncertain": "   ! язык определён неуверенно; проверьте результат или задайте язык через --language",
        "no_speech_detected": "   ! для определения языка не найдено речи; принят язык: {language}",
        "mixed": "   языки: {languages} (разговор на нескольких языках, частей: {parts})",
        "checking_parts": "   слышно больше одного языка; проверяются все части записи...",
        "scan_failed": "   ! не удалось разбить запись для определения языка ({kind}); "
                       "для всего файла используется один язык",
        # разделение по говорящим
        "separating": "   разделение по говорящим...",
        "separation_failed": "   ! разделить по говорящим не удалось ({kind}); продолжаем без этого",
        "diarize_extra": "Для разделения по говорящим нужны дополнительные пакеты: "
                         "pip install 'hebrew-transcribe[diarize]' "
                         "(из клона репозитория: bash setup.sh --diarize)",
        "diarize_cloud": "{model} работает в облаке; поддерживаются только локальные конвейеры",
        "diarize_failed": "Не удалось загрузить {model} ({kind}: {error}). Примите условия модели "
                          "на Hugging Face под той учётной записью, которой принадлежит токен, "
                          "и проверьте токен.",
        "diarize_no_access": "Нет доступа к {model}. Примите условия модели на Hugging Face под той "
                             "учётной записью, которой принадлежит токен, и проверьте токен.",
        "loading_diarizer": "Загрузка модели разделения по говорящим...",
        "diarizer_ready": "  готово: {model}",
        "without_speakers": "  Продолжаем без разделения по говорящим.",
        # токен Hugging Face
        "token_help": (
            "\nРазделение по говорящим использует модели pyannote с Hugging Face. Они\n"
            "бесплатны, но доступ к ним нужно запросить, поэтому один раз нужен\n"
            "бесплатный токен доступа:\n"
            "  1. Создайте учётную запись:  https://huggingface.co/join\n"
            "  2. Примите условия модели:   https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  3. Создайте токен «Read»:    https://huggingface.co/settings/tokens\n"
        ),
        "token_prompt": "Токен Hugging Face (ввод скрыт; Enter — пропустить): ",
        "token_saved": "  сохранён в {path} (доступен только вам; удалить: hf auth logout)",
        "token_not_saved": "  ! не удалось сохранить токен ({error}); он будет использован только в этот раз",
        # сеть
        "network_help": (
            "\n  Интернет нужен только для этого однократного скачивания. Попробуйте по порядку:\n"
            "    1. Отключитесь от рабочего VPN или смените сеть (подойдёт точка доступа\n"
            "       телефона) и запустите ту же команду снова. Скачивание продолжится\n"
            "       с того же места.\n"
            "    2. В клоне репозитория на macOS, за корпоративным прокси:\n"
            "       bash setup.sh --corp-certs\n"
        ),
        "hint_403": "  (403 Forbidden обычно означает, что скачивание блокирует прокси или межсетевой экран.)",
        # запуск
        "nothing": "Нечего расшифровывать.",
        "would_overwrite": "x {name}: пропущено — расшифровка перезаписала бы этот файл. "
                           "Укажите другую папку в --out.",
        "already_done": "- {name}: уже расшифровано (--force — расшифровать заново)",
        "file_header": "[{index}/{total}] {name}",
        "file_failed": "   x ошибка: {kind}: {error}",
        "file_skipped": "   x {error}",
        "file_done": "   готово: {name}.txt  (сегментов: {segments}, отмечено {mark}: {flagged}, {minutes} мин)",
        "finished": "Завершено. Расшифровок: {count}, папка: {path}",
        "check_marks": "Сверьте каждую расшифровку с записью, начиная с отметок {mark}.",
        "stopped": "Остановлено. Запустите ту же команду снова, чтобы продолжить с того же места.",
        "error_prefix": "x {error}",
        # справка командной строки
        "cli.description": "Приватная локальная расшифровка речи на иврите и русском из аудио "
                           "и видео, с разделением по говорящим по желанию.",
        "cli.epilog": "Перед использованием прочтите отказ от ответственности в README. "
                      "Сверяйте каждую расшифровку с исходной записью.",
        "cli.usage": "использование: ",
        "cli.positionals": "позиционные аргументы",
        "cli.options": "параметры",
        "cli.error": "{prog}: ошибка: {message}",
        "help.inputs": "аудио- или видеофайлы и/или папки",
        "help.out": "папка для результатов (по умолчанию: ./transcripts)",
        "help.recursive": "искать также во вложенных папках",
        "help.model": "'turbo' (по умолчанию), 'full' (больше и медленнее) или любая модель "
                      "faster-whisper: id на Hugging Face или локальная папка",
        "help.language": "язык записей: he (по умолчанию), ru, любой другой код языка, "
                         "'auto' — определять автоматически (в том числе когда в разговоре "
                         "несколько языков), или список, например 'he,ru', — выбирать из них",
        "help.prompt": "имена и термины, которые встречаются в записи, — для лучшего распознавания",
        "help.diarize": "разделять по говорящим (нужны пакеты [diarize] и бесплатный токен "
                        "Hugging Face, см. README)",
        "help.speakers": "число говорящих при --diarize (по умолчанию 0 — определить)",
        "help.diarization_model": "конвейер pyannote (по умолчанию community-1 для pyannote.audio 4.x)",
        "help.flag_below": "отмечать сегменты со средней лог-вероятностью ниже этой "
                           "(по умолчанию: {default}; чем выше, тем больше отметок)",
        "help.device": "по умолчанию auto (видеокарта NVIDIA, если есть, иначе процессор)",
        "help.threads": "потоки процессора (по умолчанию: все)",
        "help.force": "расшифровать заново, даже если расшифровка уже есть",
        "help.quiet": "не выводить текст во время работы",
        "help.download_only": "только скачать модель (модели) и выйти",
        "help.ui_language": "язык этих сообщений и заголовков расшифровок: en или ru "
                            "(или переменная TRANSCRIBE_UI_LANGUAGE)",
        "help.help": "показать эту справку и выйти",
        "help.version": "показать версию и выйти",
        "error.no_inputs": "укажите хотя бы один аудиофайл или папку, например:  transcribe ~/Recordings",
        "error.speakers": "--speakers: нужно 0 (определить) или больше",
        "error.language": "неизвестный код языка: {code}",
        # файлы расшифровок
        "txt.title": "Расшифровка — {file}",
        "txt.source": "Исходный файл",
        "txt.sha256": "SHA-256",
        "txt.duration": "Длительность",
        "txt.model": "Модель",
        "txt.language": "Язык",
        "txt.speakers": "Говорящие",
        "txt.created": "Создано",
        "txt.created_value": "{created}, локально, программой {tool}",
        "txt.not_separated": "не разделены",
        "txt.mixed": "несколько языков",
        "txt.warning": "Создано автоматически. Возможны неверные, пропущенные или выдуманные\n"
                       "слова и ошибки в определении говорящих. {mark} — низкая уверенность.\n"
                       "Сверяйте с исходной записью, прежде чем на что-либо опираться.",
        "txt.end": "Конец расшифровки. Сегментов: {segments}, отмечено {mark}: {flagged}.",
        "speaker": "Спикер {n}",
        "speaker_unknown": "Спикер ?",
        "index.title": "# Расшифровки",
        "index.updated": "Обновлено {date}. Создано локально программой hebrew-transcribe {version}.",
        "index.columns": "Исходный файл | Длительность | Сегменты | {mark} | Говорящие | Язык | Модель | Расшифровка",
        "index.yes": "да",
        "index.note": "**{mark} — низкая уверенность.** Прослушайте эти места, прежде чем цитировать. "
                      "Ошибки возможны и в неотмеченном тексте: машинную расшифровку нужно сверять "
                      "с записью.",
    },
}

LANGUAGE_NAMES: dict[str, dict[str, str]] = {
    "en": {
        "he": "Hebrew", "ru": "Russian", "en": "English", "ar": "Arabic", "uk": "Ukrainian",
        "am": "Amharic", "yi": "Yiddish", "fr": "French", "es": "Spanish", "de": "German",
        "it": "Italian", "pt": "Portuguese", "pl": "Polish", "ro": "Romanian", "tr": "Turkish",
        "fa": "Persian", "be": "Belarusian", "ka": "Georgian", "hy": "Armenian", "kk": "Kazakh",
        "uz": "Uzbek", "zh": "Chinese", "ja": "Japanese",
    },
    "ru": {
        "he": "иврит", "ru": "русский", "en": "английский", "ar": "арабский", "uk": "украинский",
        "am": "амхарский", "yi": "идиш", "fr": "французский", "es": "испанский", "de": "немецкий",
        "it": "итальянский", "pt": "португальский", "pl": "польский", "ro": "румынский",
        "tr": "турецкий", "fa": "персидский", "be": "белорусский", "ka": "грузинский",
        "hy": "армянский", "kk": "казахский", "uz": "узбекский", "zh": "китайский", "ja": "японский",
    },
}


def _initial_language() -> str:
    value = os.environ.get("TRANSCRIBE_UI_LANGUAGE", "").strip().lower()
    return value if value in UI_LANGUAGES else "en"


_current = _initial_language()


def set_ui_language(language: str) -> str:
    """Use 'en' or 'ru' for messages and readable files from now on. Unknown
    values fall back to English. Returns the language now in use."""
    global _current
    value = (language or "").strip().lower()
    _current = value if value in UI_LANGUAGES else "en"
    return _current


def get_ui_language() -> str:
    return _current


def t(key: str, **values: object) -> str:
    """The message `key` in the current language, filled in with `values`."""
    text = MESSAGES[_current].get(key) or MESSAGES["en"][key]
    return text.format(**values) if values else text


def language_name(code: str | None) -> str:
    if not code:
        return "?"
    return LANGUAGE_NAMES[_current].get(code) or LANGUAGE_NAMES["en"].get(code) or code


_SPEAKER = re.compile(r"Speaker (\d+)$")


def speaker_name(label: str | None) -> str:
    """'Speaker 2' in the current language. Labels are stored in English in the
    data (.json) and translated only for the readable files."""
    if not label:
        return t("speaker_unknown")
    match = _SPEAKER.match(label)
    return t("speaker", n=match.group(1)) if match else label
