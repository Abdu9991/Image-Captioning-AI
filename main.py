#importing relevant libraries
import os
import threading
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import warnings

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


def get_model_components():
    global processor, model
    if processor is not None and model is not None:
        return processor, model

    with _model_lock:
        if processor is not None and model is not None:
            return processor, model

        # Loading from FINETUNED_MODEL_PATH lets you use a custom fine-tuned checkpoint.
        processor = BlipProcessor.from_pretrained(MODEL_SOURCE)
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_SOURCE,
            low_cpu_mem_usage=True,
            torch_dtype=DTYPE,
        )
        model = model.to(DEVICE)
        model.eval()

    return processor, model


#function to generate caption for the input image
def generate_caption(image):
    #preprocessing the image
    processor_instance, model_instance = get_model_components()
    raw_image = Image.open(image).convert("RGB")
    inputs = processor_instance(raw_image, text=PROMPT, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        #generating caption using the model
        out = model_instance.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
        )

    caption = processor_instance.decode(out[0], skip_special_tokens=True)
    return caption

#defining the gradio interface
ifc = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(label="Detailed Caption"),
    title=" AI Image Captioning with BLIP",
    description=(
        "Upload an image to generate a detailed caption using the BLIP model. "
        f"Current model source: {MODEL_SOURCE}"
    ),
)

app = gr.mount_gradio_app(FastAPI(), ifc, path="/")

#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
