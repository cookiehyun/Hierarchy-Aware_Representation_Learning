"""
Projection Head for HMLC Training (WP2 - Method 1)

원본 MyResNet의 projection head 부분만 가져와서
입력을 4096(NatureLM embedding)에 맞게 수정.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP projection head: 4096 -> 1024 -> 128
    원본: Linear(2048->2048) -> ReLU -> Linear(2048->128)
    우리: Linear(4096->1024) -> ReLU -> Linear(1024->128)
    """

    def __init__(self, input_dim=4096, hidden_dim=1024, proj_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [bsz, 4096]
        Returns:
            z: [bsz, 128]  L2-normalised
        """
        return F.normalize(self.head(x), dim=1)


if __name__ == '__main__':
    model = ProjectionHead()
    x = torch.randn(32, 4096)
    z = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {z.shape}")
    print(f"Norm  : {z.norm(dim=1).mean():.4f}")  # ~1.0