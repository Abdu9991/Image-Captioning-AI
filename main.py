#importing relevant libraries
import os
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

processor = None
model = None
_model_lock = threading.Lock()


def is_qwen_vl_model(model_source):
    normalized = model_source.lower()
    return "qwen2.5-vl" in normalized or "qwen-vl" in normalized


IS_QWEN_VL = is_qwen_vl_model(MODEL_SOURCE)


def move_inputs_to_device(inputs):
    return {
        key: value.to(DEVICE) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


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


def generate_blip_caption(image_path, processor_instance, model_instance):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor_instance(raw_image, text=PROMPT, return_tensors="pt")
    inputs = move_inputs_to_device(inputs)

    with torch.no_grad():
        output = model_instance.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
        )

    return processor_instance.decode(output[0], skip_special_tokens=True)


def generate_qwen_caption(image_path, processor_instance, model_instance):
    raw_image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": PROMPT},
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
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
        )

    trimmed_output = output[:, inputs["input_ids"].shape[1]:]
    return processor_instance.batch_decode(
        trimmed_output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


#function to generate caption for the input image
def generate_caption(image):
    processor_instance, model_instance = get_model_components()
    if IS_QWEN_VL:
        return generate_qwen_caption(image, processor_instance, model_instance)
    return generate_blip_caption(image, processor_instance, model_instance)

#defining the gradio interface
ifc = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(label="Detailed Caption"),
    title="AI Image Captioning",
    description=(
        "Upload an image to generate a detailed caption using the configured vision-language model. "
        f"Current model source: {MODEL_SOURCE}"
    ),
)

app = gr.mount_gradio_app(FastAPI(), ifc, path="/")

#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
