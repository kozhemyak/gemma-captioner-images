# Serverless RunPod Image Captioning Service

Based on: https://github.com/brendanmckeag/gemma-captioner-images

This project provides a serverless runpod image captioning service using RunPod and Hugging Face's JoyCaption Alpha Two model. This service processes images/photos and generates descriptive captions or tags based on a customizable prompt.

---

## **Model Information**

- https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava
- https://github.com/fpgaminer/joycaption

---

## **Features**

- **Image Captioning**: Generate detailed captions / Booru tags captions for images.
- **Customizable Prompts**: Use a default or custom prompt for caption generation to structure output.
- **Recursive Search**: Search for images in subdirectories.
- **Resizing**: Resize large images to fit within specified dimensions (2048x2048).
- **Speed**: 5-7 seconds per image.

---

## **Requirements**

- Python 3.11+
- RunPod API Key

---

## **Getting started**

### Clone the Repository

```bash
git clone https://github.com/your-repo/image-captioner.git
cd image-captioner
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Environment Variables

Set .env file in the root directory and add the following or just define variables:

```bash
RUNPOD_ENDPOINT_ID=your-endpoint-id
RUNPOD_API_KEY=your-api-key
IMAGE_FOLDER=path/to/your/images
MAX_CONCURRENT=5
CAPTION_PROMPT=Your custom prompt here
MAX_TOKENS=512
```

## Usage

### Create RunPod Storage

### Create RunPod Serverless Endpoint

### Set and Invoke Client Script

### Clean-up


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
