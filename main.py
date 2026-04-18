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
from PIL import Image
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


def _get_int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Default to a smaller model for lower deployment memory usage.
DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-base"
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
FINETUNED_MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH")
MODEL_SOURCE = FINETUNED_MODEL_PATH or MODEL_ID

PROMPT = os.environ.get("CAPTION_PROMPT", "a detailed description of")
DEFAULT_MAX_NEW_TOKENS = 32 if RENDER_OPTIMIZED and not HAS_CUDA else 48
DEFAULT_MIN_NEW_TOKENS = 6 if RENDER_OPTIMIZED and not HAS_CUDA else 10
DEFAULT_NUM_BEAMS = 1 if RENDER_OPTIMIZED and not HAS_CUDA else 3

MAX_NEW_TOKENS = _get_int_env("CAPTION_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
MIN_NEW_TOKENS = _get_int_env("CAPTION_MIN_NEW_TOKENS", DEFAULT_MIN_NEW_TOKENS)
NUM_BEAMS = _get_int_env("CAPTION_NUM_BEAMS", DEFAULT_NUM_BEAMS)
REPETITION_PENALTY = float(os.environ.get("CAPTION_REPETITION_PENALTY", 1.15))
DEFAULT_DETAIL_LEVEL = os.environ.get("CAPTION_DETAIL_LEVEL", "Detailed").title()
NO_REPEAT_NGRAM_SIZE = _get_int_env("CAPTION_NO_REPEAT_NGRAM_SIZE", 3)
PRELOAD_MODEL_ON_STARTUP = os.environ.get("PRELOAD_MODEL_ON_STARTUP", "true").lower() == "true"

DEVICE = "cuda" if HAS_CUDA else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if RENDER_OPTIMIZED and DEVICE == "cpu":
    torch.set_num_threads(_get_int_env("TORCH_NUM_THREADS", 1))

processor = None
model = None
_model_lock = threading.Lock()
_preload_started = False

DETAIL_LEVELS = {
    "Brief": {
        "prompt": "a short caption of",
        "instruction": "Describe only what is visible in this image in one short creative natural sentence. Keep it simple, vivid, image-based, and story-like. Do not repeat words.",
        "sentence_count": 1,
        "min_new_tokens": max(4, MIN_NEW_TOKENS // 2),
        "max_new_tokens": max(16, MAX_NEW_TOKENS // 2),
        "num_beams": 1,
    },
    "Detailed": {
        "prompt": PROMPT,
        "instruction": "Describe only what is visible in this image in one creative natural sentence. Include the subject, appearance, action, setting, and clear visual context with a gentle story-like tone. Do not repeat words.",
        "sentence_count": 1,
        "min_new_tokens": max(6 if RENDER_OPTIMIZED and DEVICE == "cpu" else 8, MIN_NEW_TOKENS - 2),
        "max_new_tokens": max(24 if RENDER_OPTIMIZED and DEVICE == "cpu" else 32, MAX_NEW_TOKENS - 8),
        "num_beams": 1 if RENDER_OPTIMIZED and DEVICE == "cpu" else 2,
    },
    "Highly Detailed": {
        "prompt": "a richly detailed visual description of",
        "instruction": "Describe only what is clearly visible in this image in a rich, cinematic, story-like paragraph of two or three natural sentences. Focus on the real subject, action, setting, lighting, atmosphere, and background details. Keep the writing expressive but grounded in visible evidence, avoid artist names, website names, watermarks, or source attributions, and do not mention prompts or instructions.",
        "sentence_count": 3,
        "min_new_tokens": max(MIN_NEW_TOKENS, 16),
        "max_new_tokens": max(MAX_NEW_TOKENS, 88),
        "num_beams": NUM_BEAMS,
    },
}


def is_qwen_vl_model(model_source):
    normalized = model_source.lower()
    return "qwen2.5-vl" in normalized or "qwen-vl" in normalized


def is_blip2_model(model_source):
    return "blip2" in model_source.lower()


IS_QWEN_VL = is_qwen_vl_model(MODEL_SOURCE)
IS_BLIP2 = is_blip2_model(MODEL_SOURCE)


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
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return cleaned_text.strip(" ,.;:-")


def format_caption_output(text, detail_config):
    cleaned_text = clean_caption_text(text, detail_config)
    if not cleaned_text:
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
    return {
        "max_new_tokens": detail_config["max_new_tokens"],
        "min_new_tokens": detail_config["min_new_tokens"],
        "num_beams": detail_config["num_beams"],
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": REPETITION_PENALTY,
        "early_stopping": detail_config["num_beams"] > 1,
    }


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

    def _preload():
        try:
            get_model_components()
        except Exception as exc:
            print(f"Background model preload failed: {exc}")

    threading.Thread(target=_preload, daemon=True).start()


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

    return format_caption_output(
        processor_instance.decode(output[0], skip_special_tokens=True),
        detail_config,
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

    return format_caption_output(
        processor_instance.decode(output[0], skip_special_tokens=True),
        detail_config,
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
        selected_option = (caption_option or "All").title()
        captions = {
            detail_level: generate_single_caption(image, detail_level)
            for detail_level in DETAIL_LEVELS
        }

        if selected_option == "All":
            return "\n\n".join(
                f"{detail_level} Caption\n{captions[detail_level]}"
                for detail_level in DETAIL_LEVELS
            )

        return captions[selected_option]

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
                    caption_option = gr.Radio(
                        choices=["All", "Brief", "Detailed", "Highly Detailed"],
                        value="All",
                        label="Caption Selection",
                        info="Choose whether to generate all caption levels or only one.",
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


#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
