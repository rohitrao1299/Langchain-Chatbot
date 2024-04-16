from flask import Flask, render_template, request, send_file
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch
import io
import os

app = Flask(__name__)

# Function to generate AI-based images using Stable Diffusion
def generate_images_using_stable_diffusion(text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    prompt = text
    image = pipe(prompt).images[0]
    return image

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        text = request.form['text']
        image_output = generate_images_using_stable_diffusion(text)
        img_io = io.BytesIO()
        image_output.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)