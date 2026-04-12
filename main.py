#importing relevant libraries
import os
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

#loading the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
#function to generate caption for the input image
def generate_caption(image):
    #preprocessing the image
    raw_image = Image.open(image).convert('RGB')
    prompt = "a detailed description of"
    inputs = processor(raw_image, text=prompt, return_tensors="pt")
    with torch.no_grad():
    #generating caption using the model
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            min_new_tokens=20,
            num_beams=5,
            repetition_penalty=1.2,
        )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

#defining the gradio interface
ifc = gr.Interface(fn=generate_caption, 
                   inputs=gr.Image(type="filepath"),
                   outputs=gr.Textbox(label="Detailed Caption"),
                   title=" AI Image Captioning with BLIP",
                   description="Upload an image to generate a more detailed caption using the BLIP model.")   

app = gr.mount_gradio_app(FastAPI(), ifc, path="/")

#launching the interface
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
