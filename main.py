#importing relevant libraries
import os
import re
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
    BlipProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

warnings.filterwarnings("ignore")


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
MAX_NEW_TOKENS = _get_int_env("CAPTION_MAX_NEW_TOKENS", 48)
MIN_NEW_TOKENS = _get_int_env("CAPTION_MIN_NEW_TOKENS", 10)
NUM_BEAMS = _get_int_env("CAPTION_NUM_BEAMS", 3)
REPETITION_PENALTY = float(os.environ.get("CAPTION_REPETITION_PENALTY", 1.15))
DEFAULT_DETAIL_LEVEL = os.environ.get("CAPTION_DETAIL_LEVEL", "Detailed").title()
NO_REPEAT_NGRAM_SIZE = _get_int_env("CAPTION_NO_REPEAT_NGRAM_SIZE", 3)
PRELOAD_MODEL_ON_STARTUP = os.environ.get("PRELOAD_MODEL_ON_STARTUP", "true").lower() == "true"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

processor = None
model = None
_model_lock = threading.Lock()
_preload_started = False

DETAIL_LEVELS = {
    "Brief": {
        "prompt": "a concise description of",
        "min_new_tokens": max(4, MIN_NEW_TOKENS // 2),
        "max_new_tokens": max(16, MAX_NEW_TOKENS // 2),
    },
    "Detailed": {
        "prompt": PROMPT,
        "min_new_tokens": MIN_NEW_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
    "Highly Detailed": {
        "prompt": "an exhaustive and richly detailed description of",
        "min_new_tokens": max(MIN_NEW_TOKENS, 16),
        "max_new_tokens": max(MAX_NEW_TOKENS, 72),
    },
}


def is_qwen_vl_model(model_source):
    normalized = model_source.lower()
    return "qwen2.5-vl" in normalized or "qwen-vl" in normalized


IS_QWEN_VL = is_qwen_vl_model(MODEL_SOURCE)


def move_inputs_to_device(inputs):
    return {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def get_detail_config(detail_level):
    normalized_level = (detail_level or DEFAULT_DETAIL_LEVEL).title()
    return DETAIL_LEVELS.get(normalized_level, DETAIL_LEVELS["Detailed"])


def clean_caption_text(text):
    cleaned_text = re.sub(r"\s+'\s*", "'", text)
    cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)
    cleaned_text = re.sub(r"\b([A-Za-z]+)(?:\s*,\s*\1\b)+", r"\1", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b([A-Za-z]+)(?:\s+\1\b)+", r"\1", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return cleaned_text.strip()


def get_model_components():
    global processor, model
    if processor is not None and model is not None:
        return processor, model

    with _model_lock:
        if processor is not None and model is not None:
            return processor, model

        # Loading from FINETUNED_MODEL_PATH lets you use a custom fine-tuned checkpoint.
        if IS_QWEN_VL:
            processor = AutoProcessor.from_pretrained(MODEL_SOURCE)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_SOURCE,
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

    with torch.no_grad():
        output = model_instance.generate(
            **inputs,
            max_new_tokens=detail_config["max_new_tokens"],
            min_new_tokens=detail_config["min_new_tokens"],
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )

    return clean_caption_text(
        processor_instance.decode(output[0], skip_special_tokens=True)
    )


def generate_qwen_caption(image_path, processor_instance, model_instance, detail_config):
    raw_image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": detail_config["prompt"]},
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

    with torch.no_grad():
        output = model_instance.generate(
            **inputs,
            max_new_tokens=detail_config["max_new_tokens"],
            min_new_tokens=detail_config["min_new_tokens"],
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
        )

    trimmed_output = output[:, inputs["input_ids"].shape[1]:]
    return clean_caption_text(
        processor_instance.batch_decode(
            trimmed_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )


#function to generate caption for the input image
def generate_caption(image, detail_level):
    processor_instance, model_instance = get_model_components()
    detail_config = get_detail_config(detail_level)
    if IS_QWEN_VL:
        return generate_qwen_caption(image, processor_instance, model_instance, detail_config)
    return generate_blip_caption(image, processor_instance, model_instance, detail_config)

#defining the gradio interface
ifc = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="filepath"),
        gr.Dropdown(
            choices=list(DETAIL_LEVELS.keys()),
            value=DEFAULT_DETAIL_LEVEL if DEFAULT_DETAIL_LEVEL in DETAIL_LEVELS else "Detailed",
            label="Caption Detail Level",
        ),
    ],
    outputs=gr.Textbox(label="Detailed Caption"),
    title="AI Image Captioning",
    description=(
        "Upload a photo to get a natural-language caption, then adjust the detail level to make the result shorter, richer, or more descriptive."
    ),
)

app = gr.mount_gradio_app(FastAPI(), ifc, path="/")


@app.on_event("startup")
async def startup_event():
    preload_model_components()

#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
