import torch
import torch.nn as nn
import torch.nn.functional as F 

PATCH = 4
MLP_RATIO = 4
IMG = 32
NUM_CLASSES = 10

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()        
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 1. Non-negative kernel map
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 2. Linear attention kernel product
        kv = k.transpose(-2, -1) @ v 
        
        # 3. Normalization denominator
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        denom = (q @ k_sum) + 1e-6
        
        out = (q @ kv) / denom
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            QuickGELU(),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size=IMG, patch_size=PATCH, width=384, layers=9, heads=6, output_dim=NUM_CLASSES, mlp_ratio=MLP_RATIO):
        super().__init__()
        grid = image_size // patch_size
        n_tokens = grid * grid + 1
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(scale * torch.randn(n_tokens, width))
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.patch_embed(x)                                # [N, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)                       # [N, grid^2, width]
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)                         # [N, grid^2 + 1, width]
        x = x + self.pos_embed                         
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x[:, 0])                              
        return x @ self.proj

def model_vit(size="resnet18", num_classes=None):
    """Tour ViT: the projection maps the CLS token straight to `num_classes` logits."""
    if num_classes is None:                      # resolve live (after init_train set it)
        try:
            from train import NUM_CLASSES as num_classes
        except Exception:
            num_classes = 100
    assert size in CONFIGS, f"size must be one of {set(CONFIGS)}"
    return VisionTransformer(image_size=IMG, patch_size=PATCH, output_dim=num_classes, **CONFIGS[size])


if __name__ == "__main__":
    for s in ("resnet18", "resnet34"):
        m = model_vit(s)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        y = m(torch.zeros(2, 3, 32, 32))
        print(f"{s:9s}: {p:5.2f}M params, out {tuple(y.shape)}")