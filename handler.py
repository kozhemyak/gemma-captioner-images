#!/usr/bin/env python
# Handler for RunPod Serverless

import os
import runpod
import torch
from PIL import Image
import base64
import io
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig

# ===== USER MODIFIABLE SETTINGS =====
# Get model ID from environment variable with fallback to default
MODEL_ID = os.environ.get("MODEL_ID", "fancyfeast/llama-joycaption-alpha-two-hf-llava")

# Path to the network volume (mounted in RunPod)
NETWORK_VOLUME_PATH = os.environ.get("NETWORK_VOLUME_PATH", "/runpod-volume")
HF_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Set up Hugging Face token from environment variable 
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Configure Hugging Face cache directory
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR

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
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for outlier detection
)

# Load the model
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,  # Add quantization config
        cache_dir=HF_CACHE_DIR,  # Use the custom cache directory
        **token_param,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,  # Use the custom cache directory
        **token_param,
    )
    
    print(f"Model loaded on {device}")
except Exception as e:
    if not HF_TOKEN:
        print("ERROR: Failed to load model. This may be a gated model that requires a token.")
    raise e

print("Model and processor loaded and ready for inference")

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
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)  # Ensure pixel values match dtype
        
        # Track input length to extract only new tokens
        input_len = inputs["input_ids"].shape[-1]
        
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
        generate_ids = generate_ids[input_len:]
        
        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
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
        "max_new_tokens": 300  # Optional, defaults to 300
    }
    """
    job_input = job["input"]
    
    # Basic input validation
    if "image" not in job_input:
        return {"error": "No image provided in input"}
    
    # Get the prompt (optional, use default if not provided)
    prompt = job_input.get("prompt", "Write a long descriptive caption for this image in a formal tone.")
    max_new_tokens = job_input.get("max_new_tokens", 300)
    
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
