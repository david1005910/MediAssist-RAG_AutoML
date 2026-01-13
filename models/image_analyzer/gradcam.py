"""Grad-CAM visualization for model interpretability."""

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class GradCAM:
    """Grad-CAM implementation for CNN visualization."""

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input image tensor.
            target_class: Target class index for visualization.

        Returns:
            Heatmap as numpy array.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Generate heatmap
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def overlay(
        self,
        image_path: str,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay heatmap on original image.

        Args:
            image_path: Path to original image.
            heatmap: Grad-CAM heatmap.
            alpha: Transparency for overlay.

        Returns:
            PIL Image with heatmap overlay.
        """
        import cv2

        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * image)

        return Image.fromarray(overlay)
