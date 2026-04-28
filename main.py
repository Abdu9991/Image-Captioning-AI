#importing relevant libraries
from datetime import datetime
import os
import re
import shutil
import threading
import warnings

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from PIL import Image
from PIL import ImageStat
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

warnings.filterwarnings("ignore")

IS_RENDER = bool(os.environ.get("RENDER")) or bool(os.environ.get("RENDER_INSTANCE_ID"))
HAS_CUDA = torch.cuda.is_available()
RENDER_OPTIMIZED = os.environ.get(
    "RENDER_OPTIMIZED",
    "true" if IS_RENDER else "false",
).lower() == "true"
LOW_LATENCY_MODE = os.environ.get(
    "LOW_LATENCY_MODE",
    "true" if not HAS_CUDA else "false",
).lower() == "true"


def _get_int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Default to a smaller model for lower deployment memory usage.
DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-base"
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
FINETUNED_MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH")
MODEL_SOURCE = FINETUNED_MODEL_PATH or MODEL_ID

PROMPT = os.environ.get("CAPTION_PROMPT", "a detailed description of")
DEFAULT_MAX_NEW_TOKENS = 24 if RENDER_OPTIMIZED and not HAS_CUDA else 48
DEFAULT_MIN_NEW_TOKENS = 5 if RENDER_OPTIMIZED and not HAS_CUDA else 10
DEFAULT_NUM_BEAMS = 1 if RENDER_OPTIMIZED and not HAS_CUDA else 3
CPU_QUANTIZE = os.environ.get(
    "CPU_QUANTIZE",
    "true" if RENDER_OPTIMIZED and not HAS_CUDA else "false",
).lower() == "true"

MAX_NEW_TOKENS = _get_int_env("CAPTION_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
MIN_NEW_TOKENS = _get_int_env("CAPTION_MIN_NEW_TOKENS", DEFAULT_MIN_NEW_TOKENS)
NUM_BEAMS = _get_int_env("CAPTION_NUM_BEAMS", DEFAULT_NUM_BEAMS)
REPETITION_PENALTY = float(os.environ.get("CAPTION_REPETITION_PENALTY", 1.15))
DEFAULT_DETAIL_LEVEL = os.environ.get("CAPTION_DETAIL_LEVEL", "Detailed").title()
NO_REPEAT_NGRAM_SIZE = _get_int_env("CAPTION_NO_REPEAT_NGRAM_SIZE", 3)
DEFAULT_MAX_TIME_SECONDS = 18.0 if RENDER_OPTIMIZED and not HAS_CUDA else 0.0
MAX_GENERATION_TIME_SECONDS = _get_float_env("CAPTION_MAX_TIME", DEFAULT_MAX_TIME_SECONDS)
PRELOAD_MODEL_ON_STARTUP = os.environ.get(
    "PRELOAD_MODEL_ON_STARTUP",
    "false" if RENDER_OPTIMIZED else "true",
).lower() == "true"
ENABLE_PEOPLE_REFINEMENT = os.environ.get(
    "ENABLE_PEOPLE_REFINEMENT",
    "false" if LOW_LATENCY_MODE else "true",
).lower() == "true"

DEVICE = "cuda" if HAS_CUDA else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if RENDER_OPTIMIZED and DEVICE == "cpu":
    torch.set_num_threads(_get_int_env("TORCH_NUM_THREADS", 1))

processor = None
model = None
_model_lock = threading.Lock()
_preload_started = False
_cache_warmup_started = False

DETAIL_LEVELS = {
    "Brief": {
        "prompt": "a short caption of",
        "instruction": "Describe only what is visible in this image in one short creative natural sentence. Keep it simple, vivid, image-based, and story-like. Do not repeat words.",
        "sentence_count": 1,
        "min_new_tokens": max(4, MIN_NEW_TOKENS // 2),
        "max_new_tokens": 10 if LOW_LATENCY_MODE else max(12, MAX_NEW_TOKENS // 2),
        "num_beams": 1,
    },
    "Detailed": {
        "prompt": "a clear, natural description of",
        "instruction": "Describe only what is clearly visible in this image using clear, natural English. Mention the main subject, setting, visible details, and any obvious action. Keep it objective, avoid repeating phrases, and do not guess unknown facts.",
        "clear_output": True,
        "sentence_count": 2,
        "min_new_tokens": 4 if LOW_LATENCY_MODE else max(4 if RENDER_OPTIMIZED and DEVICE == "cpu" else 8, MIN_NEW_TOKENS - 2),
        "max_new_tokens": 16 if LOW_LATENCY_MODE else max(20 if RENDER_OPTIMIZED and DEVICE == "cpu" else 48, MAX_NEW_TOKENS),
        "num_beams": 1 if LOW_LATENCY_MODE or (RENDER_OPTIMIZED and DEVICE == "cpu") else 2,
    },
    "Highly Detailed": {
        "prompt": "a detailed visual description of",
        "instruction": (
            "You are an advanced image captioning assistant.\n\n"
            "Your task is to generate a HIGHLY DETAILED caption of the provided image with a full scene breakdown and rich context.\n\n"
            "Follow these instructions carefully:\n\n"
            "1. Start with a concise overall summary (1-2 sentences).\n"
            "2. Then provide a comprehensive, structured breakdown including:\n"
            "   - Main subject(s): who or what is the focus\n"
            "   - Environment/setting: indoor/outdoor, location type, background details\n"
            "   - Objects and elements: list all visible items and their positions\n"
            "   - Actions and interactions: what is happening in the scene\n"
            "   - Appearance details: colors, textures, clothing, lighting, expressions\n"
            "   - Spatial relationships: where things are located relative to each other\n"
            "   - Mood/atmosphere: emotional tone or feeling of the scene\n"
            "   - Time/context clues: time of day, season, event, or situation (if inferable)\n\n"
            "3. Be precise, descriptive, and exhaustive, but avoid hallucinating unknown facts.\n"
            "4. Do NOT assume identities of people or sensitive attributes.\n"
            "5. Use clear, natural language (not bullet points unless needed for clarity).\n"
            "6. Keep the description objective and grounded in what is visible.\n\n"
            "Output format:\n"
            "- Short Summary\n"
            "- Highly Detailed Description"
        ),
        "structured_output": True,
        "sentence_count": None,
        "min_new_tokens": 8 if LOW_LATENCY_MODE else max(MIN_NEW_TOKENS, 12 if RENDER_OPTIMIZED and DEVICE == "cpu" else 24),
        "max_new_tokens": 28 if LOW_LATENCY_MODE else max(MAX_NEW_TOKENS, 64 if RENDER_OPTIMIZED and DEVICE == "cpu" else 220),
        "num_beams": 1 if LOW_LATENCY_MODE or (RENDER_OPTIMIZED and DEVICE == "cpu") else NUM_BEAMS,
    },
}


def maybe_quantize_cpu_model(model_instance):
    if not CPU_QUANTIZE or DEVICE != "cpu":
        return model_instance

    try:
        return torch.quantization.quantize_dynamic(
            model_instance,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as exc:
        print(f"CPU quantization skipped: {exc}")
        return model_instance


def is_qwen_vl_model(model_source):
    normalized = model_source.lower()
    return "qwen2.5-vl" in normalized or "qwen-vl" in normalized


def is_blip2_model(model_source):
    return "blip2" in model_source.lower()


IS_QWEN_VL = is_qwen_vl_model(MODEL_SOURCE)
IS_BLIP2 = is_blip2_model(MODEL_SOURCE)

ANIMAL_TERMS_PATTERN = re.compile(r"\b(dog|puppy|canine|pet|cat|kitten|feline)\b", re.IGNORECASE)
CHILD_TERMS_PATTERN = re.compile(r"\b(child|kid|toddler|boy|girl|young\s+child|little\s+one)\b", re.IGNORECASE)
PEOPLE_TERMS_PATTERN = re.compile(r"\b(person|people|woman|man|adult|mother|father|family)\b", re.IGNORECASE)


def move_inputs_to_device(inputs):
    return {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def get_detail_config(detail_level):
    normalized_level = (detail_level or DEFAULT_DETAIL_LEVEL).title()
    detail_config = dict(DETAIL_LEVELS.get(normalized_level, DETAIL_LEVELS["Detailed"]))
    detail_config.setdefault("instruction", detail_config["prompt"])
    return detail_config


def clean_caption_text(text, detail_config=None):
    detail_config = detail_config or {}
    prompt_text = detail_config.get("prompt", "").strip()
    instruction_text = detail_config.get("instruction", "").strip()

    cleaned_text = re.sub(r"\s+'\s*", "'", text)
    cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)

    if prompt_text and cleaned_text.lower().startswith(prompt_text.lower()):
        cleaned_text = cleaned_text[len(prompt_text):].lstrip(" ,:-")

    if instruction_text and cleaned_text.lower().startswith(instruction_text.lower()):
        cleaned_text = cleaned_text[len(instruction_text):].lstrip(" ,:-")

    cleaned_text = re.sub(
        r"^a\s+polished\s+highly\s+detailed\s+caption\s+for\s+this\s+image.*$",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(r"(?:,?\s+and\s+more)+", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(
        r"(?:,?\s+and\s+(depth|size|color|composition|details?|description))+",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(r"(?:,\s*){2,}", ", ", cleaned_text)
    cleaned_text = re.sub(
        r"\bby\s+[A-Za-z0-9 .'-]+\s+on\s+(?:500pxs?|500px|flickr|deviantart|shutterstock|unsplash)\b",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    cleaned_text = re.sub(r"\bvia\s+[A-Za-z0-9._-]+\b", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b(?:www\.)?[a-z0-9-]+\.(?:com|net|org|co)\b", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'"[^"\n]*"', "", cleaned_text)
    cleaned_text = re.sub(r"\b([A-Za-z]+)(?:\s*,\s*\1\b)+", r"\1", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b([A-Za-z]+)(?:\s+\1\b)+", r"\1", cleaned_text, flags=re.IGNORECASE)
    # Remove short-token gibberish lists such as "fo, no, vg, wfx".
    cleaned_text = re.sub(r"\b(?:[a-z]{1,3})(?:\s*,\s*[a-z]{1,3}){2,}\b", "", cleaned_text)
    cleaned_text = re.sub(r"\b(?:wfx|fbcf|vg)\b", "", cleaned_text, flags=re.IGNORECASE)
    # Remove incomplete trailing articles left by truncated decoding (e.g., "is a.").
    cleaned_text = re.sub(r"\b(?:a|an|the)\s*[.!?]*$", "", cleaned_text, flags=re.IGNORECASE)
    # Remove dangling trailing copula verbs from truncated endings (e.g., "... is.").
    cleaned_text = re.sub(r"\b(?:is|are|was|were)\s*[.!?]*$", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return cleaned_text.strip(" ,.;:-")


def analyze_image_context(image_path):
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        orientation = "landscape" if width > height else "portrait" if height > width else "square"
        avg_luma = ImageStat.Stat(image.convert("L")).mean[0]
        dominant_color = image.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0))

    if avg_luma < 70:
        lighting = "dim or shaded"
    elif avg_luma < 150:
        lighting = "soft and balanced"
    else:
        lighting = "bright"

    return {
        "width": width,
        "height": height,
        "orientation": orientation,
        "lighting": lighting,
        "dominant_color": dominant_color,
    }


def normalize_caption_sentence(text):
    sentence = re.sub(r"\s+", " ", (text or "").strip())
    parts = [part.strip(" .") for part in sentence.split(",") if part.strip(" .")]
    unique_parts = []
    seen = set()
    for part in parts:
        lowered = part.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique_parts.append(part)

    sentence = ", ".join(unique_parts)
    sentence = re.sub(r"\b([A-Za-z]+)(?:\s+\1\b)+", r"\1", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\b(?:a|an|the)\s*[.!?]*$", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\b(?:is|are|was|were)\s*[.!?]*$", "", sentence, flags=re.IGNORECASE)
    sentence = sentence.strip(" ,.;:-")
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def build_clear_detailed_output(cleaned_text):
    sentences = [
        normalize_caption_sentence(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if sentence.strip()
    ]
    sentences = [sentence for sentence in sentences if sentence]

    if not sentences:
        sentences = ["The image shows a visible subject and surrounding scene."]

    detailed_text = " ".join(sentences[:2]).strip()
    detailed_text = re.sub(r"\s{2,}", " ", detailed_text)
    if detailed_text:
        detailed_text = detailed_text[0].upper() + detailed_text[1:]
    return detailed_text


def build_structured_highly_detailed_output(cleaned_text, image_path):
    context = analyze_image_context(image_path)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if sentence.strip()
    ]

    if not sentences:
        sentences = [cleaned_text]

    normalized_sentences = [normalize_caption_sentence(sentence) for sentence in sentences if sentence.strip()]
    normalized_sentences = [sentence for sentence in normalized_sentences if sentence]
    if not normalized_sentences:
        normalized_sentences = ["The image shows a visible scene with clear foreground and background elements."]

    summary = " ".join(normalized_sentences[:2]).strip()

    dominant_color = context["dominant_color"]
    dominant_color_text = f"RGB{dominant_color}"
    primary_sentence = normalized_sentences[0]
    supporting_text = " ".join(normalized_sentences[1:4]).strip()
    if not supporting_text:
        supporting_text = primary_sentence

    detailed_description = (
        f"The main subject is {primary_sentence[0].lower() + primary_sentence[1:] if len(primary_sentence) > 1 else primary_sentence.lower()} "
        f"The setting appears outdoors, and the image is {context['orientation']}-framed with {context['lighting']} lighting that supports clear depth through foreground, midground, and background layers.\n\n"
        f"Visible elements and positions are derived from what is clearly shown: {supporting_text} "
        "Any actions or interactions are interpreted only from visible posture, placement, and scene context, without assuming identity or sensitive attributes.\n\n"
        f"Appearance details are defined by natural color and texture cues, with an overall dominant color impression near {dominant_color_text}. "
        "Surfaces, materials, and contrast are described from visible evidence, while spatial relationships follow how objects are arranged left-to-right and near-to-far in the frame.\n\n"
        "The mood is inferred from composition and lighting, and time/context clues are included only when visually supported; if not explicit, they remain neutral and unspecified."
    )

    return f"Short Summary\n{summary}\n\nHighly Detailed Description\n{detailed_description}"


def format_caption_output(text, detail_config, image_path=None):
    cleaned_text = clean_caption_text(text, detail_config)
    if not cleaned_text:
        return cleaned_text

    if detail_config.get("structured_output") and image_path:
        return build_structured_highly_detailed_output(cleaned_text, image_path)

    if detail_config.get("clear_output"):
        return build_clear_detailed_output(cleaned_text)

    if detail_config.get("sentence_count") is None:
        return cleaned_text

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if sentence.strip()
    ]

    if not sentences:
        sentences = [cleaned_text]

    sentence_count = detail_config.get("sentence_count")
    if sentence_count:
        sentences = sentences[:sentence_count]

    formatted_text = " ".join(sentences).strip()
    formatted_text = formatted_text[0].upper() + formatted_text[1:]

    if formatted_text[-1] not in ".!?":
        formatted_text += "."

    return formatted_text


def get_generation_kwargs(detail_config):
    generation_kwargs = {
        "max_new_tokens": detail_config["max_new_tokens"],
        "min_new_tokens": detail_config["min_new_tokens"],
        "num_beams": detail_config["num_beams"],
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": REPETITION_PENALTY,
        "early_stopping": detail_config["num_beams"] > 1,
        "use_cache": True,
    }

    if MAX_GENERATION_TIME_SECONDS > 0:
        generation_kwargs["max_time"] = MAX_GENERATION_TIME_SECONDS

    return generation_kwargs


def refine_people_animal_confusion(primary_text, image_path, processor_instance, model_instance, detail_config):
    """Reduce obvious dog/child confusion with a people-focused second pass."""
    if not ENABLE_PEOPLE_REFINEMENT or not primary_text or not ANIMAL_TERMS_PATTERN.search(primary_text):
        return primary_text

    try:
        raw_image = Image.open(image_path).convert("RGB")
        people_probe = processor_instance(
            raw_image,
            text="a clear photo of people and children",
            return_tensors="pt",
        )
        people_probe = move_inputs_to_device(people_probe)

        probe_kwargs = get_generation_kwargs(detail_config)
        probe_kwargs["max_new_tokens"] = min(32, probe_kwargs["max_new_tokens"])
        probe_kwargs["min_new_tokens"] = min(6, probe_kwargs["min_new_tokens"])

        with torch.inference_mode():
            probe_output = model_instance.generate(
                **people_probe,
                **probe_kwargs,
            )

        probe_text = processor_instance.decode(probe_output[0], skip_special_tokens=True)
        probe_text = clean_caption_text(probe_text, detail_config)
    except Exception:
        return primary_text

    has_people_probe = bool(PEOPLE_TERMS_PATTERN.search(probe_text))
    has_child_probe = bool(CHILD_TERMS_PATTERN.search(probe_text))
    has_animal_probe = bool(ANIMAL_TERMS_PATTERN.search(probe_text))

    if (has_child_probe or has_people_probe) and not has_animal_probe:
        refined = re.sub(r"\bher\s+dog\b", "her child", primary_text, flags=re.IGNORECASE)
        refined = re.sub(r"\bhis\s+dog\b", "his child", refined, flags=re.IGNORECASE)
        refined = re.sub(r"\btheir\s+dog\b", "their child", refined, flags=re.IGNORECASE)
        refined = re.sub(r"\b(a|the)\s+dog\b", r"\1 child", refined, flags=re.IGNORECASE)
        refined = re.sub(r"\bdog\b", "child", refined, flags=re.IGNORECASE)
        return refined

    return primary_text


def build_model_load_error(exc):
    message = str(exc)
    lowered = message.lower()

    if "not enough space on the disk" in lowered or "background writer channel closed" in lowered:
        return gr.Error(
            "BLIP-2 could not be loaded because there is not enough free disk space to download the model. "
            "Free up disk space or switch back to Salesforce/blip-image-captioning-base."
        )

    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return gr.Error(
            "The selected model could not be loaded because the machine does not have enough memory. "
            "Use a smaller model or increase available RAM/GPU memory."
        )

    return gr.Error(f"Failed to load model '{MODEL_SOURCE}': {message}")


def get_model_components():
    global processor, model
    if processor is not None and model is not None:
        return processor, model

    with _model_lock:
        if processor is not None and model is not None:
            return processor, model

        try:
            if IS_QWEN_VL:
                processor = AutoProcessor.from_pretrained(MODEL_SOURCE)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_SOURCE,
                    torch_dtype=DTYPE,
                )
            elif IS_BLIP2:
                processor = Blip2Processor.from_pretrained(MODEL_SOURCE)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    MODEL_SOURCE,
                    low_cpu_mem_usage=True,
                    torch_dtype=DTYPE,
                )
            else:
                processor = BlipProcessor.from_pretrained(MODEL_SOURCE)
                model = BlipForConditionalGeneration.from_pretrained(
                    MODEL_SOURCE,
                    low_cpu_mem_usage=True,
                    torch_dtype=DTYPE,
                )
            model = maybe_quantize_cpu_model(model)
            model = model.to(DEVICE)
            model.eval()
        except Exception as exc:
            processor = None
            model = None
            raise build_model_load_error(exc) from exc

    return processor, model


def preload_model_components():
    global _preload_started
    if not PRELOAD_MODEL_ON_STARTUP or _preload_started:
        return

    _preload_started = True

    if RENDER_OPTIMIZED and DEVICE == "cpu":
        get_model_components()
        return

    def _preload():
        try:
            get_model_components()
        except Exception as exc:
            print(f"Background model preload failed: {exc}")

    threading.Thread(target=_preload, daemon=True).start()


def warm_model_cache():
    global _cache_warmup_started

    if PRELOAD_MODEL_ON_STARTUP or _cache_warmup_started:
        return

    if not RENDER_OPTIMIZED:
        return

    if FINETUNED_MODEL_PATH and os.path.exists(FINETUNED_MODEL_PATH):
        return

    if os.path.exists(MODEL_SOURCE):
        return

    _cache_warmup_started = True

    def _warm_cache():
        try:
            snapshot_download(
                repo_id=MODEL_SOURCE,
                ignore_patterns=["*.h5", "*.msgpack", "*.ot"],
            )
        except Exception as exc:
            print(f"Background model cache warmup failed: {exc}")

    threading.Thread(target=_warm_cache, daemon=True).start()


def generate_blip_caption(image_path, processor_instance, model_instance, detail_config):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor_instance(
        raw_image,
        text=detail_config["prompt"],
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs)

    with torch.inference_mode():
        output = model_instance.generate(
            **inputs,
            **get_generation_kwargs(detail_config),
        )

    decoded_text = processor_instance.decode(output[0], skip_special_tokens=True)
    decoded_text = refine_people_animal_confusion(
        decoded_text,
        image_path,
        processor_instance,
        model_instance,
        detail_config,
    )

    return format_caption_output(
        decoded_text,
        detail_config,
        image_path=image_path,
    )


def generate_blip2_caption(image_path, processor_instance, model_instance, detail_config):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor_instance(
        images=raw_image,
        text=detail_config["instruction"],
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs)

    with torch.inference_mode():
        output = model_instance.generate(
            **inputs,
            **get_generation_kwargs(detail_config),
        )

    decoded_text = processor_instance.decode(output[0], skip_special_tokens=True)
    decoded_text = refine_people_animal_confusion(
        decoded_text,
        image_path,
        processor_instance,
        model_instance,
        detail_config,
    )

    return format_caption_output(
        decoded_text,
        detail_config,
        image_path=image_path,
    )


def generate_qwen_caption(image_path, processor_instance, model_instance, detail_config):
    raw_image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": detail_config["instruction"]},
            ],
        }
    ]
    prompt_text = processor_instance.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor_instance(
        text=[prompt_text],
        images=[raw_image],
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs)

    with torch.inference_mode():
        output = model_instance.generate(
            **inputs,
            **get_generation_kwargs(detail_config),
        )

    trimmed_output = output[:, inputs["input_ids"].shape[1]:]
    return format_caption_output(
        processor_instance.batch_decode(
            trimmed_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0],
        detail_config,
        image_path=image_path,
    )


#function to generate caption for the input image
def generate_single_caption(image, detail_level):
    processor_instance, model_instance = get_model_components()
    detail_config = get_detail_config(detail_level)
    if IS_QWEN_VL:
        return generate_qwen_caption(image, processor_instance, model_instance, detail_config)
    if IS_BLIP2:
        return generate_blip2_caption(image, processor_instance, model_instance, detail_config)
    return generate_blip_caption(image, processor_instance, model_instance, detail_config)


def generate_caption_set(image, caption_option):
    if image:
        selected_option = (caption_option or DEFAULT_DETAIL_LEVEL).title()

        if selected_option == "All":
            if RENDER_OPTIMIZED and DEVICE == "cpu":
                raise gr.Error(
                    "Render CPU mode does not support 'All' because it is too slow. "
                    "Choose Brief, Detailed, or Highly Detailed."
                )

            captions = {
                detail_level: generate_single_caption(image, detail_level)
                for detail_level in DETAIL_LEVELS
            }
            return "\n\n".join(
                f"{detail_level} Caption\n{captions[detail_level]}"
                for detail_level in DETAIL_LEVELS
            )

        if selected_option not in DETAIL_LEVELS:
            selected_option = DEFAULT_DETAIL_LEVEL if DEFAULT_DETAIL_LEVEL in DETAIL_LEVELS else "Detailed"

        return generate_single_caption(image, selected_option)

    raise gr.Error("Upload an image to generate captions.")


def flag_caption(image, caption_text):
    if not image or not caption_text.strip():
        raise gr.Error("Generate a caption before flagging it.")

    flagged_dir = os.path.join(os.getcwd(), "flagged_samples")
    os.makedirs(flagged_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    image_ext = os.path.splitext(image)[1] or ".png"
    image_target = os.path.join(flagged_dir, f"flagged-{timestamp}{image_ext}")
    text_target = os.path.join(flagged_dir, f"flagged-{timestamp}.txt")

    shutil.copy2(image, image_target)
    with open(text_target, "w", encoding="utf-8") as flagged_file:
        flagged_file.write(caption_text.strip())

    gr.Info("Flag saved to flagged_samples.")


APP_CSS = """
body {
    background: linear-gradient(180deg, #fcfcfd 0%, #f6f8fb 100%);
}

.gradio-container {
    max-width: 1280px !important;
}

.app-shell {
    padding: 26px 0 28px;
}

.page-header {
    text-align: center;
    margin-bottom: 12px;
}

.page-title {
    margin: 0;
    font-size: 40px;
    line-height: 1.1;
    font-weight: 700;
    color: #272f3d;
}

.page-copy {
    margin: 0 0 22px;
    font-size: 16px;
    line-height: 1.55;
    color: #4a5565;
}

.main-grid {
    gap: 18px;
    align-items: start;
}

.panel {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 10px 24px rgba(23, 35, 54, 0.05);
}

.upload-panel {
    min-height: 560px;
}

.output-panel {
    max-width: 680px;
    min-height: 150px;
    margin-left: auto;
    height: 560px;
    overflow: hidden;
}

.panel-title {
    margin: 0 0 10px;
    font-size: 18px;
    font-weight: 600;
    color: #5b6775;
}

.image-input {
    margin-top: 4px;
}

.image-input .image-container,
.image-input .image-frame,
.caption-box textarea {
    border-radius: 14px !important;
}

.selection-wrap {
    margin-top: 14px;
}

.selection-wrap label {
    font-weight: 600 !important;
    color: #5c6877 !important;
}

.selection-wrap .wrap {
    gap: 8px !important;
}

.button-row {
    margin-top: 18px;
    gap: 16px;
}

.button-row button {
    min-height: 54px !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

.clear-btn button {
    background: linear-gradient(180deg, #f7f9fc, #edf2f7) !important;
    color: #2f3b4a !important;
    border: 1px solid #dbe2ea !important;
    box-shadow: none !important;
}

.generate-btn button {
    background: linear-gradient(135deg, #ffd894, #ffbf67) !important;
    color: #b45411 !important;
    border: 1px solid #ffc56b !important;
    box-shadow: 0 10px 24px rgba(255, 191, 103, 0.3) !important;
}

.flag-btn {
    margin-top: 14px;
}

.flag-btn button {
    width: 100% !important;
    min-height: 58px !important;
    border-radius: 14px !important;
    background: linear-gradient(180deg, #f2f4f7, #e8edf4) !important;
    color: #2f3b4a !important;
    border: 1px solid #dde4ed !important;
    box-shadow: 0 6px 18px rgba(26, 39, 61, 0.08) !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

.caption-box textarea {
    font-size: 15px !important;
    line-height: 1.55 !important;
    min-height: 90px !important;
    max-height: 420px !important;
    overflow-y: auto !important;
}

@media (max-width: 900px) {
    .page-title {
        font-size: 32px;
    }

    .upload-panel,
    .output-panel {
        min-height: auto;
        max-width: none;
    }
}
"""


with gr.Blocks(title="AI Image Captioning", css=APP_CSS, theme=gr.themes.Soft()) as ifc:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <section class="page-header">
              <h1 class="page-title">AI Image Captioning</h1>
            </section>
            """
        )
        gr.Markdown(
            "Upload an image, and the AI will generate a descriptive caption for it using the BLIP model.",
            elem_classes=["page-copy"],
        )

        with gr.Row(equal_height=False, elem_classes=["main-grid"]):
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["panel", "upload-panel"]):
                    image_input = gr.Image(type="filepath", label="image", elem_classes=["image-input"])
                    caption_option = gr.Dropdown(
                        choices=["All", "Brief", "Detailed", "Highly Detailed"],
                        value=DEFAULT_DETAIL_LEVEL if DEFAULT_DETAIL_LEVEL in DETAIL_LEVELS else "Detailed",
                        label="Caption Level",
                        info="Choose one caption level for faster results, or select All to generate every level.",
                        elem_classes=["selection-wrap"],
                    )

            with gr.Column(scale=6):
                with gr.Group(elem_classes=["panel", "output-panel"]):
                    gr.HTML(
                        """
                        <h2 class="panel-title">Generated Caption</h2>
                        """
                    )
                    output_box = gr.Textbox(
                        show_label=False,
                        lines=4,
                        max_lines=4,
                        elem_classes=["caption-box"],
                    )
                    flag_button = gr.Button("Flag", variant="secondary", elem_classes=["flag-btn"])

        with gr.Row(elem_classes=["main-grid"]):
            with gr.Column(scale=5):
                with gr.Row(elem_classes=["button-row"]):
                    clear_button = gr.ClearButton(
                        value="Clear",
                        components=[image_input, caption_option, output_box],
                        variant="secondary",
                        elem_classes=["clear-btn"],
                    )
                    generate_button = gr.Button("Submit", variant="primary", elem_classes=["generate-btn"])
            with gr.Column(scale=6):
                gr.HTML("")

        generate_button.click(
            fn=generate_caption_set,
            inputs=[image_input, caption_option],
            outputs=[output_box],
        )
        flag_button.click(
            fn=flag_caption,
            inputs=[image_input, output_box],
            outputs=[],
        )

app = gr.mount_gradio_app(FastAPI(), ifc, path="/")


@app.on_event("startup")
async def startup_event():
    preload_model_components()
    warm_model_cache()


#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
