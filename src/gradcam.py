import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CLIPGradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_gradients = None
        self.feature_map = None
        self.hooks = []
        
        # Hook into the last transformer block of the visual encoder
        # For ViT-B/32: model.visual.transformer.resblocks[-1]
        target_layer = self.model.visual.transformer.resblocks[-1]
        
        self.hooks.append(target_layer.register_forward_hook(self.save_feature_map))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradients))

    def save_feature_map(self, module, input, output):
        # Result of the block. Shape: [L, N, E] -> [50, 1, 768] (for ViT-B/32 with 224x224)
        # Permute to [N, L, E]
        self.feature_map = output.permute(1, 0, 2).detach()

    def save_gradients(self, module, grad_input, grad_output):
        # grad_output tuple. The first element is usually what we want.
        self.feature_gradients = grad_output[0].permute(1, 0, 2).detach()

    def generate_heatmap(self, image_input, text_input):
        """
        Generates Grad-CAM heatmap for a specific image and text pair.
        """
        self.model.zero_grad()
        
        # Forward pass
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (logit)
        logit = (image_features @ text_features.T).sum()
        
        # Backward pass
        logit.backward()
        
        # Grad-CAM calculation
        # self.feature_map: [1, 50, 768]
        # self.feature_gradients: [1, 50, 768]
        
        # Global Average Pooling on gradients to get weights
        weights = torch.mean(self.feature_gradients, dim=1, keepdim=True) # [1, 1, 768] ? No, dim=1 is sequence length
        # Actually for ViT, L=50 (1 class token + 7x7 patches).
        # We want to weight the CHANNELS (last dim 768).
        # So we pool over the spatial/sequence dimension (dim=1)
        weights = torch.mean(self.feature_gradients, dim=1, keepdim=True) # [1, 1, 768]
        
        # Weighted combination of feature maps
        # cam = sum(weights * feature_map)
        cam = (self.feature_map * weights).sum(dim=-1) # [1, 50]
        
        # Remove class token (index 0) and reshape to 7x7
        # ViT-B/32 has 7x7 patches
        cam = cam[:, 1:].reshape(1, 7, 7)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to image size (224x224)
        # Using PIL or cv2 or torch interpolation
        cam = cam.unsqueeze(1) # [1, 1, 7, 7]
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        return cam

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
