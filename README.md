# Typographic Adversarial Optimizer

## Project Overview
This repository contains a research-grade implementation of an **Adversarial Typographic Optimization Engine** for VLMs (specifically OpenAI's CLIP). It extends the finding that "Vision-LLMs Can Fool Themselves" by automating the search for the worst-case typographic attack using **Differential Evolution**.

## Key Features
- **Differential Evolution Optimization**: Automatically finds the optimal position, size, and color of text to fool the model.
- **Grad-CAM Interpretability**: Visualizes the attention shift from the object to the adversarial text, proving the "attention sink" mechanism.
- **Defense Ablation**: Includes Gaussian Blur and Total Variation Denoising to test defense robustness.
- **Modular Design**: Clean separation of attack, defense, and evaluation logic.

## Installation
1. Clone the repository.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the `demo.ipynb` notebook to see the full pipeline in action:
1. Downloads sample images.
2. Optimizes typographic attacks.
3. specifices Grad-CAM visualizations.
4. Generates performance plots.

## Methodology
### Attack Strategy (`TypoAttacker`)
We optimize the vector $\theta = \{x, y, size, r, g, b\}$ to minimize the ground truth confidence $P_{GT}$ and maximize the target text confidence $P_{Target}$.
$$ \mathcal{L} = P_{GT} + (1 - P_{Target}) $$

### Interpretability
We use Gradient-weighted Class Activation Mapping (Grad-CAM) on the last residual block of CLIP's Visual Transformer. This reveals that the model's attention is hijacked by the typographic text, effectively ignoring the visual content.

### Defenses
We evaluate "Blind" defenses:
- **Gaussian Blur**: Smooths out high-frequency text features.
- **TV Denoising**: Removes structured noise (text) while preserving edges.
