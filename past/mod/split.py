import torch
import torch.nn as nn

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        assert x.requires_grad, "Input tensor does not require gradient"
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # The resulting shape should be (B, C, num_patches_H, num_patches_W, patch_size, patch_size)
        # We need to permute and reshape it to (B, num_patches, patch_size_flattened)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_H, num_patches_W, C, patch_size, patch_size)
        x = x.reshape(B, -1, self.patch_size * self.patch_size * C)  # (B, num_patches, patch_size_flattened)
        
        # Ensure the output still has gradient tracking
        assert x.requires_grad, "Output tensor does not require gradient after split_patch"
        return x

# Usage example
patch_size = 8
images = torch.randn(1, 1, 32, 32, requires_grad=True)
patch_extractor = PatchExtractor(patch_size)

# Extract patches
patches = patch_extractor(images)

# Check the output shape
print("Patches shape:", patches.shape)  # Expected shape: (1, 16, 64)
print("Patches requires_grad:", patches.requires_grad)  # Should be True
