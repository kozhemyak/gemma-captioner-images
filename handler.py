#!/usr/bin/env python
# Handler for RunPod Serverless Gemma 3 Image Captioning

import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-27b-pt"

# Prompt for image captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a short, single-line description of this image for training data."
# =====================================

# Set up Hugging Face token from environment variable 
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Load the model once at startup, outside of the handler
print(f"Loading model: {MODEL_ID}")

# Configure token parameters if provided
if HF_TOKEN:
    token_param = {"token": HF_TOKEN}
    print("Using configured Hugging Face token from environment variable")
else:
    token_param = {}
    print("No Hugging Face token provided (this will only work for non-gated models)")

# Configure quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, **token_param)
    
    print(f"Model loaded on {device}")
except Exception as e:
    if not HF_TOKEN:
        print("ERROR: Failed to load model. This may be a gated model that requires a token.")
    raise e

print("Model and processor loaded and ready for inference")

def caption_image(image_data, prompt=CAPTION_PROMPT, max_new_tokens=256):
    """Generate a caption for the given image."""
    try:
        # Instead of using messages with the template, create direct inputs
        # This is a workaround for the chat template issue
        text_inputs = processor.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Process the image separately 
        image_inputs = processor.image_processor(image_data, return_tensors="pt").to(model.device)
        
        # Generate caption directly using the processed inputs
        with torch.inference_mode():
            outputs = model.generate(
                **text_inputs,
                images=image_inputs.pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode the caption, starting after the input tokens
        caption = processor.tokenizer.decode(outputs[0][text_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Ensure caption is a single line
        caption = caption.replace('\n', ' ').strip()
        return caption
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error processing image: {str(e)}\n{traceback_str}"


def handler(job):
    """
    This is the handler function that will be called by the serverless worker.
    Job input format:
    {
        "image": "base64 encoded image or URL",
        "prompt": "Optional custom prompt for captioning",
        "max_new_tokens": 256  # Optional, defaults to 256
    }
    """
    job_input = job["input"]
    
    # Basic input validation
    if "image" not in job_input:
        return {"error": "No image provided in input"}
    
    # Get the prompt (optional, use default if not provided)
    prompt = job_input.get("prompt", CAPTION_PROMPT)
    max_new_tokens = job_input.get("max_new_tokens", 256)
    
    # Handle the image (base64, URL, or file path)
    image_input = job_input["image"]
    
    try:
        # Case 1: Base64 encoded image
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            # Extract base64 part after the comma
            base64_data = image_input.split(",")[1]
            image_data = Image.open(io.BytesIO(base64.b64decode(base64_data)))
        
        # Case 2: Pure base64 string (without data URI prefix)
        elif isinstance(image_input, str) and len(image_input) > 100:
            try:
                image_data = Image.open(io.BytesIO(base64.b64decode(image_input)))
            except Exception:
                # If not a valid base64, try as URL or file path
                if image_input.startswith(('http://', 'https://')):
                    # It's a URL, we need to download it
                    import requests
                    response = requests.get(image_input, stream=True)
                    response.raise_for_status()  # Will raise an exception for HTTP errors
                    image_data = Image.open(io.BytesIO(response.content))
                else:
                    # Assume it's a file path
                    image_data = Image.open(image_input)
        
        # Case 3: URL starting with http:// or https://
        elif isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
            # It's a URL, we need to download it
            import requests
            response = requests.get(image_input, stream=True)
            response.raise_for_status()  # Will raise an exception for HTTP errors
            image_data = Image.open(io.BytesIO(response.content))
        
        # Case 4: Local file path
        elif isinstance(image_input, str):
            # Assume it's a file path
            image_data = Image.open(image_input)
        
        else:
            return {"error": "Invalid image format. Please provide a base64 encoded image, URL, or file path."}
        
        # Convert to RGB mode to ensure compatibility
        image_data = image_data.convert("RGB")
        
        # Process the image to get the caption
        caption = caption_image(image_data, prompt, max_new_tokens)
        
        # Return the result
        return {
            "caption": caption
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {"error": f"Error processing image: {str(e)}", "traceback": error_trace}


# Start the serverless function
runpod.serverless.start({"handler": handler})
