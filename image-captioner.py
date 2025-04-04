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

# ===== USER MODIFIABLE SETTINGS =====
# Your RunPod endpoint ID
ENDPOINT_ID = "ct2uzploqjakqv"

# Your RunPod API key
API_KEY = "rpa_YLYPRMRJI7AHU3LZ0GL9Z4QHNE3NW90XZNS8E5UC1w3z4y"

# Maximum concurrent requests
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "5"))

# Polling interval in seconds for async requests
POLLING_INTERVAL = int(os.environ.get("POLLING_INTERVAL", "2"))
# =====================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RunPod Image Captioning Client')
    parser.add_argument('--image_folder', type=str, required=True, 
                        help='Path to folder containing images')
    parser.add_argument('--endpoint_id', type=str, default=ENDPOINT_ID,
                        help=f'RunPod endpoint ID (default: {ENDPOINT_ID})')
    parser.add_argument('--api_key', type=str, default=API_KEY,
                        help='RunPod API key')
    parser.add_argument('--concurrent', type=int, default=MAX_CONCURRENT,
                        help='Maximum number of concurrent requests')
    parser.add_argument('--sync', action='store_true',
                        help='Use synchronous requests instead of async')
    return parser.parse_args()

def encode_image_to_base64(image_path):
    """Load an image and convert it to base64."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB mode if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            
            # Encode to base64
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return encoded_image
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def send_request_sync(image_path, args):
    """Send a synchronous request to the RunPod API."""
    try:
        image_name = os.path.basename(image_path)
        print(f"Processing {image_name}...")
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return
        
        # Prepare API request - only send the image
        url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "image": base64_image
            }
        }
        
        # Send request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Check for errors
        if "error" in result:
            print(f"Error with {image_name}: {result['error']}")
            return
        
        # Save caption
        image_base = os.path.splitext(image_path)[0]
        output_path = f"{image_base}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['output']['caption'])
        
        print(f"Caption saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def send_request_async(image_path, args):
    """Send an asynchronous request to the RunPod API and poll for results."""
    try:
        image_name = os.path.basename(image_path)
        print(f"Processing {image_name}...")
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return
        
        # Prepare API request for async operation - only send the image
        url = f"https://api.runpod.ai/v2/{args.endpoint_id}/run"
        headers = {
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": {
                "image": base64_image
            }
        }
        
        # Send request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        job_id = result.get('id')
        if not job_id:
            print(f"Error: No job ID returned for {image_name}")
            return
        
        # Poll for results
        status_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/status/{job_id}"
        
        while True:
            time.sleep(POLLING_INTERVAL)
            status_response = requests.get(status_url, headers=headers)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data.get('status')
            
            if status == 'COMPLETED':
                # Save caption
                image_base = os.path.splitext(image_path)[0]
                output_path = f"{image_base}.txt"
                
                caption = status_data.get('output', {}).get('caption')
                if caption:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    print(f"Caption saved to {output_path}")
                else:
                    print(f"No caption received for {image_name}")
                break
            
            elif status == 'FAILED':
                print(f"Job failed for {image_name}: {status_data.get('error')}")
                break
            
            elif status == 'CANCELLED':
                print(f"Job cancelled for {image_name}")
                break
            
            elif status == 'IN_QUEUE' or status == 'IN_PROGRESS':
                print(f"Job status for {image_name}: {status}")
            
            else:
                print(f"Unknown status for {image_name}: {status}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def main():
    """Main function to run the client."""
    args = parse_arguments()
    
    # Print configuration
    print(f"Using endpoint ID: {args.endpoint_id}")
    print(f"Max concurrent requests: {args.concurrent}")
    
    # Validate arguments
    if args.endpoint_id == "your-endpoint-id-here" or args.api_key == "your-runpod-api-key-here":
        print("ERROR: You must set your RunPod endpoint ID and API key")
        print("You can do this in the script, via command line arguments, or environment variables:")
        print("  RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY")
        sys.exit(1)
    
    # Get list of image files
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [
        os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)
        if os.path.splitext(f.lower())[1] in supported_formats
    ]
    
    if not image_files:
        print(f"No supported image files found in {args.image_folder}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    request_fn = send_request_sync if args.sync else send_request_async
    
    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = [executor.submit(request_fn, image_path, args) for image_path in image_files]
        for future in futures:
            future.result()  # Wait for all requests to complete
    
    print("All captions generated successfully")

if __name__ == "__main__":
    main()
