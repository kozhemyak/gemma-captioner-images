#!/usr/bin/env python
# Handler for RunPod Serverless

import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

# ===== USER MODIFIABLE SETTINGS =====
MODEL_ID = os.environ.get("MODEL_ID", "fancyfeast/llama-joycaption-alpha-two-hf-llava")
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
HF_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR

# Load the model once at startup, outside of the handler
print(f"Loading model: {MODEL_ID}")

if HF_TOKEN:
    token_param = {"token": HF_TOKEN}
else:
    token_param = {}

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=HF_CACHE_DIR,
        **token_param,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        **token_param,
    )
    
    print(f"Model loaded on {device}")
except Exception as e:
    if not HF_TOKEN:
        print("ERROR: Failed to load model. This may be a gated model that requires a token.")
    raise e

print("Model and processor loaded and ready for inference")

def log_message(message, level="INFO"):
    """Log messages with timestamp, level, and color."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
    }
    color = level_colors.get(level.upper(), Fore.WHITE)
    print(f"{Fore.CYAN}[{timestamp}] {color}[{level}] {message}{Style.RESET_ALL}")

def caption_image(image_data, prompt, max_new_tokens):
    """Generate a caption for the given image."""
    try:
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        # Format the conversation
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)
        
        # Process inputs
        inputs = processor(text=[convo_string], images=[image_data], return_tensors="pt").to(device)
        inputs = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        
        # Debugging: Log input types
        log_message(f"Input IDs dtype: {inputs['input_ids'].dtype}", level="INFO")
        log_message(f"Pixel values dtype: {inputs['pixel_values'].dtype}", level="INFO")
        
        # Generate caption
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]
        
        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[-1]:]
        
        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.replace('\n', ' ').strip()
        return caption
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        log_message(f"Error processing image: {str(e)}\n{traceback_str}", level="ERROR")
        return f"Error processing image: {str(e)}"

def handler(job):
    """
    This is the handler function that will be called by the serverless worker.
    Job input format:
    {
        "image": "base64 encoded image or URL",
        "prompt": "Optional custom prompt for captioning",
        "max_new_tokens": 300  # Optional, defaults to 300
    }
    """
    job_input = job["input"]
    
    # Basic input validation
    if "image" not in job_input:
        return {"error": "No image provided in input"}
    
    prompt = job_input.get("prompt", "Write a long descriptive caption for this image in a formal tone.")
    max_new_tokens = job_input.get("max_new_tokens", 300)
    image_input = job_input["image"]
    
    try:
        # Case 1: Base64 encoded image
        if isinstance(image_input, str) and image_input.startswith("data:image"):
            log_message("Detected Base64 image with data URI prefix.", level="INFO")
            base64_data = image_input.split(",")[1].strip()
            image_data = Image.open(io.BytesIO(base64.b64decode(base64_data)))
        
        # Case 2: Pure base64 string
        elif isinstance(image_input, str) and len(image_input) > 100:
            log_message("Detected pure Base64 image.", level="INFO")
            try:
                image_data = Image.open(io.BytesIO(base64.b64decode(image_input.strip())))
            except Exception:
                log_message("Base64 decoding failed. Trying as URL or file path.", level="WARNING")
                if image_input.startswith(('http://', 'https://')):
                    log_message("Detected URL.", level="INFO")
                    import requests
                    response = requests.get(image_input, stream=True)
                    response.raise_for_status()
                    image_data = Image.open(io.BytesIO(response.content))
                else:
                    log_message("Assuming file path.", level="INFO")
                    image_data = Image.open(image_input)
        
        # Case 3: URL starting with http:// or https://
        elif isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
            log_message("Detected URL.", level="INFO")
            import requests
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            image_data = Image.open(io.BytesIO(response.content))
        
        # Case 4: Local file path
        elif isinstance(image_input, str):
            log_message("Assuming file path.", level="INFO")
            image_data = Image.open(image_input)
        
        else:
            return {"error": "Invalid image format. Please provide a base64 encoded image, URL, or file path."}
        
        # Convert to RGB mode to ensure compatibility
        if image_data.mode != "RGB":
            log_message("Converting image to RGB.", level="INFO")
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
        log_message(f"Error processing image: {str(e)}\n{error_trace}", level="ERROR")
        return {"error": f"Error processing image: {str(e)}", "traceback": error_trace}

# Start the serverless function
runpod.serverless.start({"handler": handler})
