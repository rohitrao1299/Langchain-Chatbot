import streamlit as st 
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

# Function to generate AI-based images using Stable Diffusion
def generate_images_using_stable_diffusion(text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    prompt = text
    image = pipe(prompt).images[0]
    return image

# Streamlit Code
st.title("AI Image Generation App")
st.subheader("Image generation using Stable Diffusion")

input_prompt = st.text_input("Enter your text prompt")
if input_prompt is not None:
    if st.button("Generate Image"):
        image_output = generate_images_using_stable_diffusion(input_prompt)
        st.info("Generating image.....")
        st.success("Image Generated Successfully")
        st.image(image_output, caption="Generated by Stable Diffusion")