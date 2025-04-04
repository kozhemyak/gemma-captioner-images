# gemma-captioner: Quick and easy video and image captioning with Gemma 3

These are one shot scripts designed to work with Gemma 3 by Google that automatically performs captioning on an entire folder of images, and will create caption .txt files with the same file name as the corresponding image or video. Best used in a fine-tuning pipeline for open source video packages like Mochi, Wan, LTX, or Hunyuan Video (the latter three best accomplished through diffusion-pipe.) It will automatically download the model listed in the settings.

## Image captioning

Usage:
```
python gemma3-image-captioning.py --image_folder images/  
```
Iterates over an folder of images and provides a caption.  

Edit the arguments at the top of the .py to your liking.

Intended for use on RunPod serverless - prompt and model are controlled through environment variables. 
![image](https://github.com/user-attachments/assets/f6ab3def-2b45-4981-acdf-379a4cb93031)
