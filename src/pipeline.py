import pandas as pd
import torch
import os
from src.utils import load_image, load_clip_model, get_probs, process_image_for_clip
from src.attacker import TypoAttacker
from src.defense import defense_pipeline
import requests
from io import BytesIO
from PIL import Image

class EvaluationPipeline:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = load_clip_model(self.device)
        self.attacker = TypoAttacker(self.model, self.preprocess, self.device)
        self.results = []
        
    def download_image(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Debug prints
            # print(f"DEBUG: Status {response.status_code}, Type: {response.headers.get('content-type')}, Size: {len(response.content)}")
            
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Failed to download {url}")
            print(f"Error: {e}")
            if 'response' in locals():
                print(f"Status: {response.status_code}")
                print(f"Content-Type: {response.headers.get('content-type')}")
                print(f"Content sample: {response.content[:20]}")
            return None

    def run_benchmark(self, image_data):
        """
        image_data: list of dicts {'url': ..., 'ground_truth': ..., 'target_text': ...}
        """
        for i, item in enumerate(image_data):
            print(f"Processing Image {i+1}/{len(image_data)}...")
            
            # Load Image
            if 'url' in item:
                original_image = self.download_image(item['url'])
            elif 'path' in item:
                original_image = load_image(item['path'])
            else:
                continue
                
            if original_image is None:
                continue

            # Resize for consistency/speed (CLIP expects 224, but we can operate on larger)
            # Keeping original aspect ratio or resizing to reasonable max dimension
            original_image.thumbnail((512, 512))
            
            gt_class = item['ground_truth']
            target_text = item['target_text']
            
            # Baseline Confidence
            img_tensor = process_image_for_clip(original_image, self.preprocess, self.device)
            base_probs = get_probs(self.model, img_tensor, [gt_class, target_text], self.device)
            orig_conf_gt = base_probs[0]
            
            # Run Optimization
            adv_image, best_params, final_loss = self.attacker.optimize_attack(
                original_image, target_text, gt_class
            )
            
            # Attack Confidence
            adv_tensor = process_image_for_clip(adv_image, self.preprocess, self.device)
            adv_probs = get_probs(self.model, adv_tensor, [gt_class, target_text], self.device)
            adv_conf_gt = adv_probs[0]
            adv_conf_target = adv_probs[1]
            
            conf_drop = (orig_conf_gt - adv_conf_gt) * 100
            
            # Defenses
            # 1. Blur
            blurred_img = defense_pipeline(adv_image, method="blur", device=self.device)
            blur_tensor = process_image_for_clip(blurred_img, self.preprocess, self.device)
            blur_probs = get_probs(self.model, blur_tensor, [gt_class, target_text], self.device)
            blur_conf_gt = blur_probs[0]
            
            # 2. TV Denoising
            tv_img = defense_pipeline(adv_image, method="tv", device=self.device)
            tv_tensor = process_image_for_clip(tv_img, self.preprocess, self.device)
            tv_probs = get_probs(self.model, tv_tensor, [gt_class, target_text], self.device)
            tv_conf_gt = tv_probs[0]

            self.results.append({
                "Image_ID": i,
                "Ground_Truth": gt_class,
                "Target_Text": target_text,
                "Original_Conf": orig_conf_gt,
                "Attacked_Conf": adv_conf_gt,
                "Attacked_Target_Conf": adv_conf_target,
                "Confidence_Drop_Pct": conf_drop,
                "Blur_Defense_Conf": blur_conf_gt,
                "TV_Defense_Conf": tv_conf_gt,
                "Best_Params": best_params
            })
            
            # Save images for later visualization
            os.makedirs("results", exist_ok=True)
            original_image.save(f"results/img_{i}_original.png")
            adv_image.save(f"results/img_{i}_attacked.png")
            
        return pd.DataFrame(self.results)
