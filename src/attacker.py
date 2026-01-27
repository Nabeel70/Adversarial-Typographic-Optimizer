import numpy as np
import torch
import clip
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import differential_evolution
import os

class TypoAttacker:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.font_path = "arial.ttf" # Default font, might need adjustment based on system

    def _render_text(self, image, params):
        """
        Renders text onto the image based on parameters.
        params: [x, y, font_size, r, g, b] (normalized or raw?) 
        Let's assume params are raw from the optimizer but need casting.
        """
        # Unpack parameters
        # DE works with floats, so we cast to int
        x, y, font_size, r, g, b = params
        
        x = int(x)
        y = int(y)
        font_size = int(font_size)
        color = (int(r), int(g), int(b))
        
        # Create a copy to draw on
        adv_image = image.copy()
        draw = ImageDraw.Draw(adv_image)
        
        try:
            # Try to load a font, fall back to default if fails
            # On Windows, arial.ttf is usually available.
            # We can try/except this.
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((x, y), self.target_text, fill=color, font=font)
        
        return adv_image

    def _objective_function(self, params, image, ground_truth_class, target_text):
        """
        Objective function for differential evolution.
        Goal: Minimize GT confidence, Maximize Target confidence.
        """
        # 1. Render image with current params
        # We need to pass target_text here or store it? 
        # Better to store it in self temporarily or pass partial.
        # But 'params' is the only thing DE changes.
        # So we set self.target_text before calling DE or use 'args' in DE.
        
        # Wait, DE allows 'args' to be passed to the func.
        
        # Re-unpack args if passed via DE args mechanism, but let's stick to using helper
        # Wrapper will handle the args.
        
        # Ensure params are within valid ranges (DE handles bounds, but casting is needed)
        
        adv_image = self._render_text(image, params)
        
        # 2. Process for CLIP
        image_input = self.preprocess(adv_image).unsqueeze(0).to(self.device)
        
        # 3. Get probabilities
        text_list = [ground_truth_class, target_text]
        text_tokenized = clip.tokenize(text_list).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.model(image_input, text_tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
        prob_gt = probs[0]
        prob_target = probs[1]
        
        # Loss: We want low prob_gt and high prob_target.
        # Minimize: prob_gt + (1 - prob_target)
        loss = prob_gt + (1.0 - prob_target)
        
        return loss

    def optimize_attack(self, image, target_text, ground_truth_class):
        """
        Runs the differential evolution optimization.
        """
        self.target_text = target_text # Store for render access
        
        width, height = image.size
        
        # Bounds:
        # x: [0, width]
        # y: [0, height]
        # font_size: [10, 100]
        # r, g, b: [0, 255]
        bounds = [
            (0, width),
            (0, height),
            (10, 100),
            (0, 255),
            (0, 255),
            (0, 255)
        ]
        
        # Run DE
        result = differential_evolution(
            self._objective_function,
            bounds,
            args=(image, ground_truth_class, target_text),
            strategy='best1bin',
            maxiter=10, # Limiting for speed in demo, can be increased
            popsize=10,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        best_params = result.x
        final_image = self._render_text(image, best_params)
        final_loss = result.fun
        
        return final_image, best_params, final_loss

