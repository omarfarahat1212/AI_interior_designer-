from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import pipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler ,DDIMScheduler,EulerAncestralDiscreteScheduler
import requests
import io
import cv2

# Define the request body
class ImageRequest(BaseModel):
    image_url: str
    prompt: str
    negative_prompt: str
    schedule_type: str
    guidance_scale: int
    strength: float
    steps: int
    seed: int
    threshold1: float
    threshold2: float
    ai_slider: float

app = FastAPI()

# Load the model once when the application starts
@app.on_event("startup")
async def load_model():
    global pipe, controlnet
    # Load the ControlNet model for Canny edge detection
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the standard 1.5 Stable Diffusion model
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "/www/wwwroot/models/XSarchitectural-InteriorDesign-ForXSLora/",
    controlnet=controlnet,
    control_mode="prompt",  # Specify the control mode here
    torch_dtype=torch.float16).to(device)

# Function to get the edge map using Canny edge detection
def get_edge_map(image, threshold1, threshold2):
    image = image.convert("L")  # Convert to grayscale
    image = np.array(image)
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    edges = np.stack([edges, edges, edges], axis=2)
    edge_map = torch.from_numpy(edges).float() / 255.0
    edge_map = edge_map.permute(2, 0, 1)
    return edge_map

@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    try:
        if request.schedule_type == 'DDIM':
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        pipe.enable_model_cpu_offload()
    
        response = requests.get(request.image_url)
        room_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Get original dimensions
        original_width, original_height = room_image.size

        # Initialize the edge detection pipeline
        edge_map = get_edge_map(room_image, request.threshold1, request.threshold2)
        
        # Convert room image to tensor
        room_image = np.array(room_image)
        room_image = torch.from_numpy(room_image).float() / 255.0
        room_image = room_image.permute(2, 0, 1).unsqueeze(0).half().to("cuda")
        
        # Resize the edge map to match the image size
        edge_map = edge_map.unsqueeze(0)
        edge_map = torch.nn.functional.interpolate(edge_map, size=(original_height, original_width)).half().to("cuda")
        
        # Set seed for reproducibility
        seed = request.seed
        generator = torch.manual_seed(seed)
        control_weight = 2
        generated_image = pipe(prompt=request.prompt, negative_prompt=request.negative_prompt, image=room_image, control_image=edge_map, guidance_scale=7, generator=generator,
        strength=0.75,num_inference_steps=request.steps,control_weight=control_weight).images[0]
        
        # Convert the generated image to base64
        buffered = BytesIO()
        generated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"image": img_str}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
