import argparse
from pathlib import Path
import json
import warnings
from typing import List, Dict, Any, Union
from httpx import Response as HttpxBinaryResponseContent
import yaml
import typst
from pdf2image import convert_from_path
from moviepy import VideoFileClip, ImageClip, vfx
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.AudioClip import CompositeAudioClip

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
) -> Dict[str, Any]:
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
                this_physical_slides[start_from]["speeches"].append(item["v"]["body"])
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


def gen_speech(
    speeches: List[str],
) -> List[Dict[str, Union[str, float, AudioFileClip]]]:
    if CONFIG["tts_tool"] == "openai":
        return gen_speech_openai(speeches)
    elif CONFIG["tts_tool"] == "fasterqwen":
        return gen_speech_fasterqwentts(speeches)
    elif CONFIG["tts_tool"] == "load":
        return gen_speech_load(speeches)
    else:
        raise ValueError(f"Unknown TTS tool: {CONFIG['tts_tool']}")


def gen_speech_fasterqwentts(
    speeches: List[str],
) -> List[Dict[str, Union[str, float, AudioFileClip]]]:
    """Generate speech using a local FasterQWen TTS server.

    The function expects configuration under ``CONFIG['fasterqwen']`` with
    optional keys:

    * ``api``: base URL of the TTS server (default ``http://localhost:7860``).
    * ``model_id``: model identifier to load (default ``models/Qwen3-TTS-12Hz-1.7B-Base``).
    * ``mode``: generation mode (``voice_clone``/``custom``/``voice_design``;
       defaults to ``voice_clone``).
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

    # ensure model is loaded before generating any samples
    resp = requests.post(f"{base_url}/load", data={"model_id": model_id})
    try:
        print("FasterQWen load response:", resp.json())
    except Exception:  # pragma: no cover - defensive
        print("FasterQWen load failed with status", resp.status_code)
        resp.raise_for_status()

    speech_data: List[Dict[str, Union[str, float, AudioFileClip]]] = []
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

        data: Dict[str, Union[str, float]] = {"text": speech, "mode": mode}
        # propagate optional parameters from config
        cfg = CONFIG.get("fasterqwen", {})
        if cfg.get("speaker"):
            data["speaker"] = cfg["speaker"]
        if cfg.get("instruct"):
            data["instruct"] = cfg["instruct"]

        files = {}
        if mode == "voice_clone" and cfg.get("ref_audio"):
            files["ref_audio"] = open(cfg["ref_audio"], "rb")

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


def gen_speech_openai(
    speeches: List[str],
) -> List[Dict[str, Union[str, float, AudioFileClip]]]:
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
    speeches: List[str],
) -> List[Dict[str, Union[str, float, AudioFileClip]]]:
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


def slides_to_images(file: Path, dpi=200, skip_saving=False) -> List[Path]:
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
    physical_slide_to_speech: List[Dict[str, Union[str, float]]],
    physical_slide_images: List[str],
    speech_data: List[Dict[str, Union[str, float, AudioFileClip]]],
    typst_root_dir: Path = None,
    transition={
        "duration": 0.8,
        "type": "fade",
    },
    audio_gap=0.2,
    size=(1920, 1080),
) -> List[Dict[str, Union[str, float]]]:
    """This will compose the video clip
    @param physical_slide_to_speech: List of physical slides
    @param speech_data: List of speech data
    @param transition_time: Time for transition between physical slides
    @param audio_gap: Gap between each audio for speech
    """

    # See https://zulko.github.io/moviepy/user_guide/compositing.html

    def dimension_to_absolute(dimension: Union[int, float, str], reference: int) -> int:
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
            print(f"Adding into {physical_slide_i}: {speech}")
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
        if physical_slide_i > 0:
            if transition["type"] == "fade":
                print("Adding transition")
                image_clip = image_clip.with_effects(
                    [vfx.CrossFadeIn(transition["duration"])]
                )
        image_clip = image_clip.with_start(time_played)
        video_clips.append(image_clip)

        # Advance the time to the next slide's
        time_played += this_duration

    composite_video = CompositeVideoClip(video_clips)
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
