# gemma-captioner: Quick and easy video and image captioning with Gemma 3

These are one shot scripts designed to work with Gemma 3 by Google that automatically performs captioning on an entire folder of images, and will create caption .txt files with the same file name as the corresponding image or video. Best used in a fine-tuning pipeline for open source video packages like Mochi, Wan, LTX, or Hunyuan Video (the latter three best accomplished through diffusion-pipe.) It will automatically download the model listed in the settings.

## Image captioning

Usage:
```
python gemma3-image-captioning.py --image_folder images/ --hf_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Iterates over an folder of images and provides a caption. hf_token optional, but required if you are trying to download a gated model.

Edit the arguments at the top of the .py to your liking:
```
# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-27b-pt"

# Prompt for image captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a short, single-line description of this image for training data."
# =====================================
```
