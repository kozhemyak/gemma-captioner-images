#!/usr/bin/env python
# Client script to send local images to RunPod endpoint and save captions

import os
import sys
import time
import base64
import argparse
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from colorama import init, Fore, Style

# Инициализация colorama
init(autoreset=True)

# ===== USER MODIFIABLE SETTINGS =====
ENDPOINT_ID         = "YOUR_ENDPOINT_ID"
API_KEY             = "YOUR_API_KEY"
MAX_CONCURRENT      = int(os.environ.get("MAX_CONCURRENT", "1"))
POLLING_INTERVAL    = int(os.environ.get("POLLING_INTERVAL", "2"))
MAX_TOKENS          = int(os.environ.get("MAX_TOKENS", "512"))
CAPTION_PROMPT      = os.environ.get("CAPTION_PROMPT", "Write a descriptive caption for this image in a formal tone.")

# =====================================

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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RunPod Image Captioning Client')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--endpoint_id', type=str, default=ENDPOINT_ID, help=f'RunPod endpoint ID')
    parser.add_argument('--api_key', type=str, default=API_KEY, help='RunPod API key')
    parser.add_argument('--concurrent', type=int, default=MAX_CONCURRENT, help='Maximum number of concurrent requests')
    parser.add_argument('--caption_prompt', type=str, default=CAPTION_PROMPT, help='Prompt for image captioning')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Maximum tokens to generate')
    parser.add_argument('--recurse', action='store_true', help='Recursively search for images in subdirectories')
    return parser.parse_args()

def find_images_in_directory(directory, recurse=False):
    """Find all supported image files in the directory and optionally its subdirectories."""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []

    if recurse:
        log_message(f"Recursively searching for images in {directory}...", level="INFO")
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file.lower())[1] in supported_formats:
                    image_files.append(os.path.join(root, file))
    else:
        log_message(f"Searching for images in {directory}...", level="INFO")
        for file in os.listdir(directory):
            if os.path.splitext(file.lower())[1] in supported_formats:
                image_files.append(os.path.join(directory, file))

    log_message(f"Found {len(image_files)} images.", level="INFO")
    return image_files

def encode_image_to_base64(image_path):
    """Load an image and convert it to base64 in PNG format."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB mode if needed
            if img.mode != "RGB":
                log_message(f"Converting image {image_path} to RGB...", level="INFO")
                img = img.convert("RGB")
            
            # Save to bytes buffer in PNG format
            buffer = BytesIO()
            img.save(buffer, format="PNG")  # Use PNG to avoid JPEG compression artifacts
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8').strip()
            log_message(f"Successfully encoded image {image_path} to Base64.", level="INFO")
            return encoded_image
    except Exception as e:
        log_message(f"Error encoding image {image_path}: {str(e)}", level="ERROR")
        return None

def send_request_sync(image_path, args):
    """Send a synchronous request to the RunPod API."""
    try:
        image_name = os.path.basename(image_path)
        log_message(f"Processing {image_name}...", level="INFO")
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            log_message(f"Skipping {image_name} due to encoding error.", level="ERROR")
            return
        
        # Prepare API request
        url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "image": base64_image,
                "prompt": args.caption_prompt,
                "max_tokens": args.max_tokens
            }
        }
        
        # Send request
        log_message(f"Sending request for {image_name}...", level="INFO")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Check for errors
        if "error" in result:
            log_message(f"Error with {image_name}: {result['error']}", level="ERROR")
            return
        
        # Save caption
        image_base = os.path.splitext(image_path)[0]
        output_path = f"{image_base}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['output']['caption'])
        
        log_message(f"Caption saved to {output_path}", level="INFO")
        
    except Exception as e:
        log_message(f"Error processing {image_path}: {str(e)}", level="ERROR")

def main():
    """Main function to run the client."""
    args = parse_arguments()
    
    # Print configuration
    log_message(f"Using endpoint ID: {args.endpoint_id}", level="INFO")
    log_message(f"Max concurrent requests: {args.concurrent}", level="INFO")
    log_message(f"Recursive search enabled: {args.recurse}", level="INFO")
    
    # Validate arguments
    if args.endpoint_id == "your-endpoint-id-here" or args.api_key == "your-runpod-api-key-here":
        log_message("ERROR: You must set your RunPod endpoint ID and API key", level="ERROR")
        log_message("You can do this in the script, via command line arguments, or environment variables:", level="ERROR")
        log_message("  RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", level="ERROR")
        sys.exit(1)
    
    # Find image files
    image_files = find_images_in_directory(args.image_folder, recurse=args.recurse)
    
    if not image_files:
        log_message(f"No supported image files found in {args.image_folder}", level="ERROR")
        sys.exit(1)
    
    # Process images
    request_fn = send_request_sync
    
    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = [executor.submit(request_fn, image_path, args) for image_path in image_files]
        for future in futures:
            future.result()  # Wait for all requests to complete
    
    log_message("All captions generated successfully", level="INFO")

if __name__ == "__main__":
    main()
