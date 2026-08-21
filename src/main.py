import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import typst
import yaml
from httpx import Response as HttpxBinaryResponseContent
from moviepy import ImageClip, VideoFileClip, vfx
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from pdf2image import convert_from_path

# from moviepy.multithreading import multithread_write_videofile

TMP_DIR = "./tmp"
CONFIG = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-c", "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-f", "--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Codec to use for the output video, if you want to speed up, use hevc_nvenc or h264_nvenc",
    )
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1080)

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(".mp4")

    return args


def query(
    file: Path,
) -> dict[str, Any]:
    J = json.loads(typst.query(file, "<t2s-file>", field="value", one=True))

    #####################
    # Defaults
    #####################
    if "t2sdefaults" in J:
        defaults = J["t2sdefaults"]
    else:
        defaults = {
            "duration_physical": 2,
            "transition": "none",
            "transition_duration": 0,
        }

    #########################
    # Logical slide settings
    #########################
    logical_slide_to_speech = []
    physical_count = []
    for physical_slide in J["pages"]:
        if physical_slide["hidden"]:
            # Simply ignore hidden slides
            continue
        if physical_slide["overlay"] == 0:  # First physical slide in a logical slide
            logical_slide_to_speech.append(physical_slide["t2s"])
            physical_count.append(1)
        else:
            # Overlay slide
            physical_count[-1] += 1

    #########################
    # Physical slide calculations
    #########################
    physical_slide_to_speech = []
    for logical_slide, physical_count in zip(logical_slide_to_speech, physical_count):
        this_physical_slides = []
        for _ in range(physical_count):
            this_physical_slides.append(
                {
                    "speeches": [],
                    "video-overlays": [],
                    "duration": defaults["duration_physical"],
                }
            )
        is_non_defaut_duration_set = False
        for item in logical_slide:
            if item["t"] == "T2s":
                start_from = max(item["v"]["start_from"] - 1, 0)
                assert (
                    start_from < physical_count
                ), f"Start from {start_from + 1} is more than the number of physical slides {physical_count} in the logical slide {logical_slide}"
                speaker_id = int(item["v"].get("speaker_id", 0) or 0)
                body = item["v"].get("body")
                language = item["v"].get("language")
                this_physical_slides[start_from]["speeches"].append(
                    {"body": body, "speaker_id": speaker_id, "language": language}
                )
            elif item["t"] == "T2s-duration-logical":
                assert (
                    not is_non_defaut_duration_set
                ), "Multiple duration settings for the same logical slide"
                is_non_defaut_duration_set = True
                for i in range(len(this_physical_slides)):
                    this_physical_slides[i]["duration"] = item["v"] / physical_count
            elif item["t"] == "T2s-duration-physical":
                assert (
                    not is_non_defaut_duration_set
                ), "Multiple duration settings for the same logical slide"
                is_non_defaut_duration_set = True
                durations = list(item["v"])
                if len(durations) < physical_count:
                    if len(durations) != 1:
                        warnings.warn(
                            f"The number of durations does not match the number of physical slides in the logical slide...  Assuming all remaining slides have the last duration, happened in {item}"
                        )
                    durations += [durations[-1]] * (physical_count - len(durations))
                elif len(durations) > physical_count:
                    warnings.warn(
                        f"The number of durations is more than the number of physical slides in the logical slide...  Ignoring the extra durations, happened in {item}"
                    )
                    durations = durations[:physical_count]
                for i, duration in enumerate(durations):
                    this_physical_slides[i]["duration"] = duration
            elif item["t"] == "T2s-video-overlay":
                start_from = max(item["v"]["start_from"] - 1, 0)
                assert (
                    start_from < physical_count
                ), f"Start from {start_from + 1} is more than the number of physical slides {physical_count} in the logical slide {logical_slide}"
                this_physical_slides[start_from]["video-overlays"].append(item["v"])

            else:
                raise ValueError(f"Unknown type: {item['t']}")

        physical_slide_to_speech += this_physical_slides

    return {
        "defaults": defaults,
        "physical_slide_to_speech": physical_slide_to_speech,
        "logical_slide_to_speech": logical_slide_to_speech,
    }


def _normalize_speech_entry(
    speech: str | dict[str, Any] | None
) -> dict[str, Any]:
    if isinstance(speech, dict):
        body = speech.get("body") or speech.get("text") or ""
        speaker_id = int(speech.get("speaker_id", 0) or 0)
        language = speech.get("language")
    elif speech is None:
        body = ""
        speaker_id = 0
        language = None
    else:
        body = str(speech)
        speaker_id = 0
        language = None
    return {"body": body, "speaker_id": speaker_id, "language": language}


def _warn_red(message: str):
    # ANSI escape code for red text in most terminals
    print(f"\033[31m{message}\033[0m", file=sys.stderr)
    # warnings.warn(message)


def _info_blue(message: str):
    # ANSI escape code for blue text in most terminals
    print(f"\033[34m{message}\033[0m", file=sys.stderr)


def _effective_speech_language(speech: dict[str, Any]) -> str | None:
    language = speech.get("language")
    if CONFIG is None:
        return language

    if CONFIG.get("tts_tool") == "fasterqwen":
        return language or CONFIG.get("fasterqwen", {}).get("language", "Auto")

    if CONFIG.get("tts_tool") == "indextts":
        language = language or CONFIG.get("indextts", {}).get("lang_choice", "ZH")
        return {
            "Chinese": "ZH",
            "English": "EN",
            "Japanese": "JA",
            "Arabic": "AR",
            "Spanish": "ES",
        }.get(language, language)

    return language


def _validate_fasterqwen_config() -> None:
    if CONFIG is None or CONFIG.get("tts_tool") != "fasterqwen":
        return

    cfg = CONFIG.get("fasterqwen", {})
    errors = []

    def check_string_list(key: str, require_exists: bool = False):
        value = cfg.get(key)
        if value is None:
            return

        if isinstance(value, list):
            if len(value) == 0:
                errors.append(f"{key} must be a nonempty list")
                return
            for idx, item in enumerate(value):
                if not isinstance(item, str) or item.strip() == "":
                    errors.append(f"{key}[{idx}] is empty or not a string")
                elif require_exists:
                    path = Path(item)
                    if not path.exists():
                        errors.append(f"{key}[{idx}] path does not exist: {item}")
            return

        if isinstance(value, str):
            if value.strip() == "":
                errors.append(f"{key} must be a nonempty string")
            elif require_exists:
                path = Path(value)
                if not path.exists():
                    errors.append(f"{key} path does not exist: {value}")
            return

        errors.append(f"{key} must be a string or list of strings")

    # check speaker / instruct list elements
    check_string_list("speaker")
    check_string_list("instruct")

    # ref_audio path validity (reuse check_string_list with filesystem check)
    check_string_list("ref_audio", require_exists=True)

    valid_languages = {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
    language = cfg.get("language", "Auto")
    if not isinstance(language, str) or language.strip() == "":
        errors.append("language must be a nonempty string")
    elif language not in valid_languages:
        errors.append(
            "language must be one of: " + ", ".join(sorted(valid_languages))
        )

    if errors:
        print("\n[ERROR] FasterQWen TTS config validation failed:")
        for msg in errors:
            print("  -", msg)
        print(
            "Please fix config.yaml (fasterqwen section) and retry."
        )
        sys.exit(1)


def _validate_indextts_config() -> None:
    if CONFIG is None or CONFIG.get("tts_tool") != "indextts":
        return

    cfg = CONFIG.get("indextts", {})
    errors = []

    api = cfg.get("api", "http://localhost:7860")
    if not isinstance(api, str) or api.strip() == "":
        errors.append("api must be a nonempty string")

    ref_audio = cfg.get("ref_audio")
    ref_audio_paths = ref_audio if isinstance(ref_audio, list) else [ref_audio]
    if not ref_audio_paths or any(
        not isinstance(path, str) or path.strip() == "" for path in ref_audio_paths
    ):
        errors.append("ref_audio must be a nonempty string or list of strings")
    else:
        for index, path in enumerate(ref_audio_paths):
            if not Path(path).exists():
                label = f"ref_audio[{index}]" if isinstance(ref_audio, list) else "ref_audio"
                errors.append(f"{label} path does not exist: {path}")

    emo_ref_path = cfg.get("emo_ref_path")
    if emo_ref_path is not None:
        if not isinstance(emo_ref_path, str) or emo_ref_path.strip() == "":
            errors.append("emo_ref_path must be a nonempty string when configured")
        elif not Path(emo_ref_path).exists():
            errors.append(f"emo_ref_path path does not exist: {emo_ref_path}")

    valid_languages = {"ZH", "EN", "JA", "AR", "ES"}
    if cfg.get("lang_choice", "ZH") not in valid_languages:
        errors.append("lang_choice must be one of ZH, EN, JA, AR, ES")

    valid_emo_methods = {
        "与音色参考音频相同",
        "使用情感参考音频",
        "使用情感向量控制",
    }
    if cfg.get("emo_control_method", "与音色参考音频相同") not in valid_emo_methods:
        errors.append("emo_control_method is not a supported IndexTTS value")

    numeric_ranges = {
        "emo_weight": (0, 1),
        "top_p": (0, 1),
        "temperature": (0, None),
        "duration_factor": (0, None),
        "top_k": (0, None),
        "num_beams": (1, None),
        "max_mel_tokens": (1, None),
        "max_text_tokens_per_segment": (1, None),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        value = cfg.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{key} must be a number")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            bound = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            errors.append(f"{key} must be in range {bound}")

    if errors:
        print("\n[ERROR] IndexTTS config validation failed:")
        for message in errors:
            print("  -", message)
        print("Please fix config.yaml (indextts section) and retry.")
        sys.exit(1)


def gen_speech(
    speeches: list[str | dict[str, Any]],
) -> list[dict[str, str | float | AudioFileClip]]:
    if CONFIG["tts_tool"] == "openai":
        normalized = [_normalize_speech_entry(s)["body"] for s in speeches]
        return gen_speech_openai(normalized)
    elif CONFIG["tts_tool"] == "fasterqwen":
        normalized = [_normalize_speech_entry(s) for s in speeches]
        return gen_speech_fasterqwentts(normalized)
    elif CONFIG["tts_tool"] == "indextts":
        normalized = [_normalize_speech_entry(s) for s in speeches]
        return gen_speech_indextts(normalized)
    elif CONFIG["tts_tool"] == "load":
        normalized = [_normalize_speech_entry(s)["body"] for s in speeches]
        return gen_speech_load(normalized)
    else:
        raise ValueError(f"Unknown TTS tool: {CONFIG['tts_tool']}")


def gen_speech_fasterqwentts(
    speeches: list[str],
) -> list[dict[str, str | float | AudioFileClip]]:
    """Generate speech using a local FasterQWen TTS server.

    The function expects configuration under ``CONFIG['fasterqwen']`` with
    optional keys:

    * ``api``: base URL of the TTS server (default ``http://localhost:7860``).
    * ``model_id``: model identifier to load (default ``models/Qwen3-TTS-12Hz-1.7B-Base``).
    * ``mode``: generation mode (``voice_clone``/``custom``/``voice_design``;
       defaults to ``voice_clone``).
     * ``language``: language passed to the TTS server (default ``Auto``).
    * ``speaker``: when using ``custom`` mode, the speaker ID.
    * ``instruct``: when using ``voice_design`` mode, the instruction text.
    * ``ref_audio``: path to a reference audio file for ``voice_clone`` mode.

    The API is called in the same manner as the snippet provided by the user:
    loading the model once and then posting ``/generate`` for each speech.
    The base64-encoded waveform is decoded, saved under ``TMP_DIR`` and
    wrapped in a :class:`moviepy.audio.io.AudioFileClip.AudioFileClip`.
    """
    import base64

    import requests

    base_url = CONFIG.get("fasterqwen", {}).get("api", "http://localhost:7860")
    model_id = CONFIG.get("fasterqwen", {}).get(
        "model_id", "models/Qwen3-TTS-12Hz-1.7B-Base"
    )
    mode = CONFIG.get("fasterqwen", {}).get("mode", "voice_clone")
    language = CONFIG.get("fasterqwen", {}).get("language", "Auto")

    # ensure model is loaded before generating any samples
    resp = requests.post(f"{base_url}/load", data={"model_id": model_id})
    try:
        print("FasterQWen load response:", resp.json())
    except Exception:  # pragma: no cover - defensive
        print("FasterQWen load failed with status", resp.status_code)
        resp.raise_for_status()

    speech_data: list[dict[str, str | float | AudioFileClip]] = []
    for i, speech_entry in enumerate(speeches):
        if not isinstance(speech_entry, dict):
            speech_entry = _normalize_speech_entry(speech_entry)

        text = speech_entry.get("body", "")
        speaker_id = max(0, int(speech_entry.get("speaker_id", 0) or 0))
        speech_language = speech_entry.get("language") or language

        print(
            f"Generating speech {i+1}/{len(speeches)}: {text} "
            f"(speaker_id={speaker_id}, language={speech_language})"
        )
        if text is None or len(str(text).strip()) == 0:
            speech_data.append(
                {
                    "file": None,
                    "duration": 0,
                    "audio_clip": None,
                }
            )
            continue

        data: dict[str, str | float] = {
            "text": str(text),
            "language": speech_language,
            "mode": mode,
        }
        cfg = CONFIG.get("fasterqwen", {})

        files: dict[str, Any] = {}

        if mode == "custom":
            raw_speaker = cfg.get("speaker")
            if isinstance(raw_speaker, list):
                if speaker_id < len(raw_speaker):
                    data["speaker"] = raw_speaker[speaker_id]
                    _info_blue(
                        f"speaker_id {speaker_id} maps to speaker '{raw_speaker[speaker_id]}'"
                    )
                else:
                    used_speaker_id = len(raw_speaker) - 1
                    _warn_red(
                        f"speaker_id {speaker_id} out of range for speaker list (len={len(raw_speaker)}), "
                        f"using index {used_speaker_id} ({raw_speaker[used_speaker_id]})"
                    )
                    data["speaker"] = raw_speaker[used_speaker_id]
            elif isinstance(raw_speaker, str):
                if speaker_id > 0:
                    _warn_red(
                        f"speaker_id {speaker_id} is > 0 but speaker is configured as a single string; "
                        f"treating as fallback value ({raw_speaker})"
                    )
                else:
                    _info_blue(
                        f"speaker_id {speaker_id} maps to speaker '{raw_speaker}'"
                    )
                data["speaker"] = raw_speaker

        elif mode == "voice_design":
            raw_instruct = cfg.get("instruct")
            if isinstance(raw_instruct, list):
                if speaker_id < len(raw_instruct):
                    data["instruct"] = raw_instruct[speaker_id]
                    _info_blue(
                        f"speaker_id {speaker_id} maps to instruct '{raw_instruct[speaker_id]}'"
                    )
                else:
                    used_instruct_id = len(raw_instruct) - 1
                    _warn_red(
                        f"speaker_id {speaker_id} out of range for instruct list (len={len(raw_instruct)}), "
                        f"using index {used_instruct_id} ({raw_instruct[used_instruct_id]})"
                    )
                    data["instruct"] = raw_instruct[used_instruct_id]
            elif isinstance(raw_instruct, str):
                if speaker_id > 0:
                    _warn_red(
                        f"speaker_id {speaker_id} is > 0 but instruct is configured as a single string; "
                        f"treating as fallback value ({raw_instruct})"
                    )
                else:
                    _info_blue(
                        f"speaker_id {speaker_id} maps to instruct '{raw_instruct}'"
                    )
                data["instruct"] = raw_instruct

        elif mode == "voice_clone":
            raw_ref_audio = cfg.get("ref_audio")
            if isinstance(raw_ref_audio, list) and len(raw_ref_audio) > 0:
                if speaker_id < len(raw_ref_audio):
                    ref_audio_path = raw_ref_audio[speaker_id]
                    _info_blue(
                        f"speaker_id {speaker_id} maps to ref_audio '{ref_audio_path}'"
                    )
                else:
                    used_ref_audio_id = len(raw_ref_audio) - 1
                    _warn_red(
                        f"speaker_id {speaker_id} out of range for ref_audio list (len={len(raw_ref_audio)}), "
                        f"using index {used_ref_audio_id} ({raw_ref_audio[used_ref_audio_id]})"
                    )
                    ref_audio_path = raw_ref_audio[used_ref_audio_id]
                files["ref_audio"] = open(ref_audio_path, "rb")
            elif isinstance(raw_ref_audio, str):
                if speaker_id > 0:
                    _warn_red(
                        f"speaker_id {speaker_id} is > 0 but ref_audio is configured as a single string; "
                        f"treating as fallback value ({raw_ref_audio})"
                    )
                else:
                    _info_blue(
                        f"speaker_id {speaker_id} maps to ref_audio '{raw_ref_audio}'"
                    )
                files["ref_audio"] = open(raw_ref_audio, "rb")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        r = requests.post(f"{base_url}/generate", data=data, files=files or None)
        r.raise_for_status()
        result = r.json()

        wav_b64 = result.get("audio_b64")
        if wav_b64 is None:
            raise RuntimeError(f"unexpected response: {result}")
        wav_bytes = base64.b64decode(wav_b64)
        out_path = Path(f"{TMP_DIR}/speech_{i}.wav")
        with open(out_path, "wb") as f:
            f.write(wav_bytes)

        audio_clip = AudioFileClip(str(out_path))
        speech_data.append(
            {
                "file": None,
                "duration": audio_clip.duration,
                "audio_clip": audio_clip,
            }
        )
    return speech_data


def gen_speech_indextts(
    speeches: list[dict[str, Any]],
) -> list[dict[str, str | float | AudioFileClip]]:
    """Generate speech through an IndexTTS Gradio server."""
    import shutil
    from urllib.request import urlretrieve

    from gradio_client import Client, handle_file

    cfg = CONFIG.get("indextts", {})
    client = Client(cfg.get("api", "http://localhost:7860"))
    ref_audio = cfg.get("ref_audio")
    if not ref_audio:
        raise ValueError("indextts.ref_audio must be configured")
    language_codes = {
        "Chinese": "ZH",
        "English": "EN",
        "Japanese": "JA",
        "Arabic": "AR",
        "Spanish": "ES",
    }

    def select_reference(speaker_id: int) -> str:
        if isinstance(ref_audio, list):
            if not ref_audio:
                raise ValueError("indextts.ref_audio must be a nonempty list")
            index = min(speaker_id, len(ref_audio) - 1)
            if index != speaker_id:
                _warn_red(
                    f"speaker_id {speaker_id} out of range for ref_audio list (len={len(ref_audio)}), "
                    f"using index {index} ({ref_audio[index]})"
                )
            return str(ref_audio[index])
        return str(ref_audio)

    speech_data: list[dict[str, str | float | AudioFileClip]] = []
    for i, speech_entry in enumerate(speeches):
        text = speech_entry.get("body", "")
        speaker_id = max(0, int(speech_entry.get("speaker_id", 0) or 0))
        language = speech_entry.get("language") or cfg.get("lang_choice", "ZH")
        language = language_codes.get(language, language)
        print(
            f"Generating speech {i+1}/{len(speeches)}: {text} "
            f"(speaker_id={speaker_id}, language={language})"
        )
        if text is None or len(str(text).strip()) == 0:
            speech_data.append({"file": None, "duration": 0, "audio_clip": None})
            continue

        reference = select_reference(speaker_id)
        result = client.predict(
            emo_control_method=cfg.get("emo_control_method", "与音色参考音频相同"),
            prompt=handle_file(reference),
            text=str(text),
            lang_choice=language,
            emo_ref_path=handle_file(cfg.get("emo_ref_path", reference)),
            emo_weight=cfg.get("emo_weight", 0.65),
            vec1=cfg.get("vec1", 0.0),
            vec2=cfg.get("vec2", 0.0),
            vec3=cfg.get("vec3", 0.0),
            vec4=cfg.get("vec4", 0.0),
            vec5=cfg.get("vec5", 0.0),
            vec6=cfg.get("vec6", 0.0),
            vec7=cfg.get("vec7", 0.0),
            vec8=cfg.get("vec8", 0.0),
            emo_text=cfg.get("emo_text", ""),
            emo_random=cfg.get("emo_random", False),
            max_text_tokens_per_segment=cfg.get("max_text_tokens_per_segment", 120),
            duration_factor=cfg.get("duration_factor", 1.0),
            param_18=cfg.get("do_sample", True),
            param_19=cfg.get("top_p", 0.8),
            param_20=cfg.get("top_k", 30),
            param_21=cfg.get("temperature", 0.8),
            param_22=cfg.get("length_penalty", 0.0),
            param_23=cfg.get("num_beams", 3),
            param_24=cfg.get("repetition_penalty", 10.0),
            param_25=cfg.get("max_mel_tokens", 1500),
            api_name="/gen_single",
        )
        response = result
        if isinstance(response, (list, tuple)) and len(response) == 1:
            response = response[0]
        source_url = None
        if isinstance(response, dict):
            result = (
                response.get("path")
                or response.get("name")
                or response.get("value")
            )
            source_url = response.get("url")
        else:
            result = response
        if not isinstance(result, (str, Path)) and not isinstance(source_url, str):
            raise RuntimeError(
                f"unexpected IndexTTS /gen_single response: {response!r}"
            )
        if isinstance(result, (str, Path)):
            source_path = Path(result)
            out_path = Path(TMP_DIR) / f"speech_{i}{source_path.suffix or '.wav'}"
            shutil.copyfile(source_path, out_path)
        else:
            out_path = Path(TMP_DIR) / f"speech_{i}.wav"
            urlretrieve(source_url, out_path)
        audio_clip = AudioFileClip(str(out_path))
        speech_data.append(
            {"file": str(out_path), "duration": audio_clip.duration, "audio_clip": audio_clip}
        )
    return speech_data


def gen_speech_openai(
    speeches: list[str],
) -> list[dict[str, str | float | AudioFileClip]]:
    from openai import OpenAI

    with open(CONFIG["openai"]["api_key"]) as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)

    speech_data = []
    for i, speech in enumerate(speeches):
        print(f"Generating speech {i+1}/{len(speeches)}: {speech}")
        if speech is None or len(speech.strip()) == 0:
            speech_data.append(
                {
                    "file": None,
                    "duration": 0,
                    "audio_clip": None,
                }
            )
            continue
        audio_response: HttpxBinaryResponseContent = client.audio.speech.create(
            input=speech,
            model=CONFIG["openai"]["model"],
            voice=CONFIG["openai"]["voice"],
            response_format="wav",
            speed=CONFIG["openai"]["speed"],
        )
        with open(f"{TMP_DIR}/speech_{i}.wav", "wb") as f:
            f.write(audio_response.content)
        audio_clip = AudioFileClip(f"{TMP_DIR}/speech_{i}.wav")
        speech_data.append(
            {
                "file": None,
                "duration": audio_clip.duration,
                "audio_clip": audio_clip,
            }
        )
    return speech_data


def gen_speech_load(
    speeches: list[str],
) -> list[dict[str, str | float | AudioFileClip]]:
    """
    This function will load the speech from the given path.
    """
    speech_data = []
    root = Path(CONFIG["load"]["path"])
    for i in range(len(speeches)):
        print(f"Loading speech {i+1}/{len(speeches)}: {speeches[i]}")
        if speeches[i] is None or len(speeches[i].strip()) == 0:
            speeches[i] = None
            speech_data.append(
                {
                    "file": None,
                    "duration": 0,
                    "audio_clip": None,
                }
            )
        else:
            fpath = root / f'speech_{i}.wav'
            if not fpath.exists():
                raise ValueError(f"Speech file {fpath} does not exist")
            audio_clip = AudioFileClip(fpath)
            speech_data.append(
                {
                    "file": str(fpath),
                    "duration": audio_clip.duration,
                    "audio_clip": audio_clip,
                }
            )
    return speech_data


def slides_to_images(file: Path, dpi=200, skip_saving=False) -> list[Path]:
    if not skip_saving:
        images = convert_from_path(file, dpi=dpi)
    else:
        images = [0] * 10
    image_files = []
    for i, image in enumerate(images):
        image_file = f"{TMP_DIR}/slide_{i}.png"
        if not skip_saving:
            image.save(image_file)
        image_files.append(image_file)
    return image_files


def compose_video_clip(
    physical_slide_to_speech: list[dict[str, str | float]],
    physical_slide_images: list[str],
    speech_data: list[dict[str, str | float | AudioFileClip]],
    typst_root_dir: Path | None = None,
    transition=None,
    audio_gap=0.2,
    size=(1920, 1080),
) -> list[dict[str, str | float]]:
    """This will compose the video clip
    @param physical_slide_to_speech: List of physical slides
    @param speech_data: List of speech data
    @param transition_time: Time for transition between physical slides
    @param audio_gap: Gap between each audio for speech
    """

    # See https://zulko.github.io/moviepy/user_guide/compositing.html

    if transition is None:
        transition = {"duration": 0.8, "type": "fade"}
    def dimension_to_absolute(dimension: float | str, reference: int) -> int:
        if dimension is None:
            return None
        if isinstance(dimension, str):
            if dimension.endswith("%"):
                return int(reference * float(dimension[:-1]) / 100)
            else:
                dimension = float(dimension)
        return int(dimension)

    if transition["type"] == "none" and transition["duration"] != 0:
        raise ValueError("Transition type is none, but duration is not 0")

    still_requied = 0  # Time to finish current audios
    time_played = 0  # The time already played until now, excluding the
    # tailing frames for each slide for transtition
    audio_started = 0  # Number of audio that has started playing

    video_clips = []
    audio_clips = []
    for physical_slide_i, (physical_slide, physical_slide_img) in enumerate(
        zip(physical_slide_to_speech, physical_slide_images)
    ):
        this_duration = physical_slide["duration"]

        # Play the video overlay
        overlay_length = 0
        for video_overlay in physical_slide["video-overlays"]:
            start_from = max(video_overlay["start_from"] - 1, 0)
            assert start_from < len(
                physical_slide_to_speech
            ), f"Start from {start_from + 1} is more than the number of physical slides {len(physical_slide_to_speech)} in the logical slide {physical_slide}"
            video_overlay_clip = VideoFileClip(typst_root_dir / video_overlay["video"])
            video_overlay_clip = video_overlay_clip.with_start(time_played)
            video_overlay_clip = video_overlay_clip.with_layer_index(2)
            video_overlay_clip = video_overlay_clip.with_position(
                (
                    dimension_to_absolute(video_overlay["x"], size[0]),
                    dimension_to_absolute(video_overlay["y"], size[1]),
                )
            )
            print(f"Adding video overlay {video_overlay} at {physical_slide_i}")
            w = dimension_to_absolute(video_overlay["width"], size[0])
            h = dimension_to_absolute(video_overlay["height"], size[1])
            if (w is not None and w > 0) or (h is not None and h > 0):
                param = {}
                if w > 0:
                    param["width"] = w
                if h > 0:
                    param["height"] = h
                print(f"Resizing video overlay to {param}")
                video_overlay_clip = video_overlay_clip.resized(**param)
            if video_overlay["reverse"]:
                video_overlay_clip = video_overlay_clip.with_effects(vfx.TimeMirror)
            video_clips.append(video_overlay_clip)
            overlay_length = max(overlay_length, video_overlay_clip.duration)
        this_duration = max(this_duration, overlay_length)

        # Play the audio
        total_audio_time_in_this_slide = 0
        for speech in physical_slide["speeches"]:
            assert (
                still_requied == 0
            ), f"Audios are not finished at slide {physical_slide_i}"
            audio_clip = speech_data[audio_started]["audio_clip"]
            speech_for_log = dict(speech)
            speech_for_log["language"] = _effective_speech_language(speech)
            print(f"Adding into {physical_slide_i}: {speech_for_log}")
            if audio_clip is None:
                # Empty audio, legit use as placeholder
                audio_started += 1
                continue
            audio_clip = audio_clip.with_start(
                time_played + total_audio_time_in_this_slide
            )
            audio_clips.append(audio_clip)
            total_audio_time_in_this_slide += audio_clip.duration + audio_gap
            audio_started += 1
        still_requied += total_audio_time_in_this_slide

        # Enlength the duration if this is the last slide for an audio
        if (
            physical_slide_i == len(physical_slide_to_speech) - 1
            or len(physical_slide_to_speech[physical_slide_i + 1]["speeches"]) > 0
        ):
            this_duration = max(this_duration, still_requied + audio_gap)
        still_requied = max(0, still_requied - this_duration)

        # Show the frame
        image_clip = ImageClip(
            physical_slide_img, duration=this_duration + transition["duration"]
        )
        image_clip = image_clip.resized(size)
        if physical_slide_i > 0 and transition["type"] == "fade":
            print("Adding transition")
            image_clip = image_clip.with_effects(
                [vfx.CrossFadeIn(transition["duration"])]
            )
        image_clip = image_clip.with_start(time_played)
        video_clips.append(image_clip)

        # Advance the time to the next slide's
        time_played += this_duration

    # Filter out invalid clips to prevent MoviePy mask broadcast errors (e.g. zero-sized or zero-duration clips).
    valid_video_clips = []
    for clip in video_clips:
        if clip is None:
            continue
        clip_width = getattr(clip, "w", None)
        clip_height = getattr(clip, "h", None)
        clip_duration = getattr(clip, "duration", None)
        if (
            clip_width is None
            or clip_height is None
            or clip_duration is None
            or clip_width <= 0
            or clip_height <= 0
            or clip_duration <= 0
        ):
            print(
                f"Skipping invalid clip in composite: width={clip_width} height={clip_height} duration={clip_duration}"
            )
            continue
        valid_video_clips.append(clip)

    if not valid_video_clips:
        raise ValueError("No valid video clips to composite")

    composite_video = CompositeVideoClip(valid_video_clips, size=size)
    composite_audio = CompositeAudioClip(audio_clips)
    composite_video = composite_video.with_audio(composite_audio)
    return composite_video


def main():
    args = parse_args()

    global CONFIG
    # force utf-8 encoding when reading config to avoid locale-based failures on Windows
    # some config files may contain characters (e.g. curly quotes) that are invalid in
    # the system default encoding (GBK on Chinese Windows).  Read as utf-8 and fall
    # back to the locale encoding if necessary.
    try:
        config_text = args.config.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        config_text = args.config.read_text(encoding=None)  # use locale fallback
    CONFIG = yaml.safe_load(config_text)

    # Create temporary directory
    Path(TMP_DIR).mkdir(exist_ok=True)

    _validate_fasterqwen_config()
    _validate_indextts_config()

    query_results = query(args.input)
    speech_texts = [
        speech
        for physical_slide in query_results["physical_slide_to_speech"]
        for speech in physical_slide["speeches"]
    ]
    print("Generating Voice-over")
    # speech_data = []
    speech_data = gen_speech(speech_texts)
    print("Converting slide to images")
    physical_slide_images = slides_to_images(
        args.input.with_suffix(".pdf"), dpi=args.dpi, skip_saving=False
    )
    print("Composing Video")
    video_clip = compose_video_clip(
        query_results["physical_slide_to_speech"],
        physical_slide_images,
        speech_data,
        typst_root_dir=args.input.parent,
        transition={
            "duration": query_results["defaults"]["transition_duration"],
            "type": query_results["defaults"]["transition"],
        },
        size=(args.height, args.width),
    )
    video_clip.write_videofile(
        args.output,
        fps=args.fps,
        codec=args.codec,
        threads=32,
        # ffmpeg_param=[
        #     "-hwaccel",
        #     "cuvid",
        # ],
    )
    # multithread_write_videofile(video_clip, args.output, fps=args.fps, codec=args.codec)


if __name__ == "__main__":
    main()
