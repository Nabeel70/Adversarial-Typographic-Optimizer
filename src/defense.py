import torch
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np

def apply_gaussian_blur(image, kernel_size=5, sigma=2.0):
    """
    Applies Gaussian Blur to a PIL Image.
    """
    blurrer = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blurrer(image)

def apply_tv_denoising(image_tensor, weight=0.1, iter_n=50):
    """
    Applies Total Variation Denoising using gradient descent on the image tensor.
    image_tensor: [1, 3, H, W] cuda tensor
    Returns: Denoised tensor
    """
    # Clone and require grad
    img = image_tensor.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([img], lr=0.01)
    
    orig = image_tensor.detach()
    
    for _ in range(iter_n):
        optimizer.zero_grad()
        
        # Fidelity term
        loss_mse = F.mse_loss(img, orig)
        
        # TV term
        # Sum of abs differences of horizontal and vertical neighbors
        loss_tv = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
                  torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        
        total_loss = loss_mse + weight * loss_tv
        total_loss.backward()
        optimizer.step()
        
    return img.detach()

def defense_pipeline(image, method="blur", device="cpu", preprocess=None):
    """
    Wrapper for defenses. 
    image: PIL Image
    """
    if method == "blur":
        return apply_gaussian_blur(image)
    elif method == "tv":
        # Convert PIL to Tensor (0-1), Denoise, Convert back to PIL.
        to_tensor = T.ToTensor()
        to_pil = T.ToPILImage()
        
        t_img = to_tensor(image).unsqueeze(0).to(device)
        denoised_t = apply_tv_denoising(t_img, weight=0.01, iter_n=50) # Light TV
        
        return to_pil(denoised_t.squeeze(0).cpu())
        
    return image
