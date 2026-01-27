import torch
import clip
from PIL import Image
import numpy as np

def load_clip_model(device):
    """
    Loads the CLIP model and preprocessing function.
    Using 'ViT-B/32' as requested (User mentioned openai/clip-vit-base-patch32 which maps to ViT-B/32 in standard clip repo).
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_image(path):
    """
    Loads an image from a path or URL (if we add URL support later).
    Returns a PIL Image.
    """
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None

def process_image_for_clip(image, preprocess, device):
    """
    Preprocesses a PIL image for CLIP.
    Returns a tensor with batch dimension.
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    return image_input

def get_probs(model, image_input, text_list, device):
    """
    Returns probabilities for the given text list.
    text_list: list of strings.
    """
    text_tokenized = clip.tokenize(text_list).to(device)
    
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image_input, text_tokenized)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
    return probs[0]
