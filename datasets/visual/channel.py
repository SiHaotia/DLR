import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_feature_from_pth(pth_path, key=None, device="cpu"):
    """
    从 .pth 文件中读取特征:
    - 如果保存的是 tensor，直接返回；
    - 如果保存的是 dict，需要指定 key，例如 key="feat"。
    返回形状为 (B, C, H, W) 的 tensor。
    """
    ckpt = torch.load(pth_path, map_location=device)

    if isinstance(ckpt, dict):
        if key is None:
            raise ValueError(
                f"{pth_path} is a dict, please specify key, e.g., key='feat'"
            )
        feat = ckpt[key]
    else:
        feat = ckpt

    if feat.dim() == 3:
        # (C, H, W) -> (1, C, H, W)
        feat = feat.unsqueeze(0)

    assert feat.dim() == 4, f"Expect 4-D tensor (B,C,H,W), but got {feat.shape}"
    return feat


# def visualize_feature_32(feat, idx_b=0, nrow=8, cmap="jet", save_path=None):
#     """
#     feat: torch.Tensor, (B, C, H, W)
#     idx_b: 选择第几个 batch 做可视化
#     nrow: 网格每行放多少个通道 (8 -> 4x8)
#     """
#     B, C, H, W = feat.shape
#     if C != 32:
#         raise ValueError(f"Expect C=32, but got C={C}")
#
#     # 取出第 idx_b 个样本
#     feat_b = feat[idx_b]              # (32, H, W)
#
#     # (32,H,W) -> (32,1,H,W)，方便 make_grid
#     feat_b = feat_b.unsqueeze(1)      # (32,1,H,W)
#
#     # 全局归一化到 [0,1]
#     f_min = feat_b.min()
#     f_max = feat_b.max()
#     feat_norm = (feat_b - f_min) / (f_max - f_min + 1e-8)
#
#     # 拼网格 (1, H_grid, W_grid)
#     grid = vutils.make_grid(feat_norm, nrow=nrow, padding=2)
#     grid_np = grid.permute(1, 2, 0).cpu().numpy()  # (H_grid, W_grid, 3)
#
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(grid_np, cmap=cmap)
#     plt.axis("off")
#     plt.tight_layout(pad=0.01)
#
#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
#         print(f"saved to {save_path}")
#     else:
#         plt.show()

# def visualize_feature_32(feat, idx_b=0, nrow=8, cmap="jet", save_path=None):
#     B, C, H, W = feat.shape
#     if C != 32:
#         raise ValueError(f"Expect C=32, but got C={C}")
#
#     feat_b = feat[idx_b]              # (32, H, W)
#     feat_b = feat_b.unsqueeze(1)      # (32, 1, H, W)
#
#     # 全局归一化到 [0,1]
#     f_min = feat_b.min()
#     f_max = feat_b.max()
#     feat_norm = (feat_b - f_min) / (f_max - f_min + 1e-8)
#
#     # 拼网格，注意这里输出是 (3, H_grid, W_grid)
#     grid = vutils.make_grid(feat_norm, nrow=nrow, padding=2)
#     grid_np = grid.permute(1, 2, 0).cpu().numpy()   # (H_grid, W_grid, 3)
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(grid_np)    # 3 通道图就不要再传 cmap 了
#     plt.axis("off")
#     plt.tight_layout(pad=0.01)
#
#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
#         print(f"saved to {save_path}")
#     else:
#         plt.show()

def visualize_feature_32(feat, idx_b=0, nrow=8, cmap_name="jet", save_path=None):
    """
    feat: torch.Tensor, shape (B, C, H, W)
    idx_b: 可视化第几个 batch 样本
    nrow: 每行显示多少个通道 (8 -> 4x8)
    cmap_name: 伪彩色 colormap 名称，例如 'jet', 'viridis', 'turbo'
    """
    B, C, H, W = feat.shape
    if C != 32:
        print(f"[Warn] C={C}, not 32, but still visualize all channels.")

    # 取第 idx_b 个样本，(C, H, W)
    feat_b = feat[idx_b]      # (C, H, W)

    # 获取 colormap 函数
    cmap = plt.get_cmap(cmap_name)

    colored_list = []
    for c in range(C):
        ch = feat_b[c].detach().cpu().numpy().astype(np.float32)  # (H, W)

        # 通道内归一化到 [0,1]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
        else:
            ch_norm = np.zeros_like(ch, dtype=np.float32)

        # 应用 colormap -> (H, W, 4)，取前 3 个通道 (RGB)
        ch_color = cmap(ch_norm)[..., :3]          # (H, W, 3), float32 in [0,1]
        ch_color = (ch_color * 255).astype(np.uint8)

        # 转成 torch (3, H, W)
        ch_tensor = torch.from_numpy(ch_color).permute(2, 0, 1)  # (3,H,W)
        colored_list.append(ch_tensor)

    # 堆成 (C, 3, H, W)
    colored_feat = torch.stack(colored_list, dim=0)  # (C,3,H,W)
    print(colored_feat.shape)
    # 用 make_grid 拼成 1 张大图
    # grid = vutils.make_grid(colored_feat, nrow=nrow, padding=2)  # (3, H_grid, W_grid)
    # grid_np = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # (H_grid, W_grid, 3)

    color_dependent = colored_feat[13]
    color_dependent = color_dependent.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    img_rgb = Image.fromarray(color_dependent, mode="RGB")
    img_rgb.save("./channelout/channel_dependent.png")

    # if save_path is not None:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    #     print(f"saved to {save_path}")
    # else:
    #     plt.show()


# def visualize_feature_32(feat, idx_b=0, nrow=8, cmap="jet", save_path=None):
#     """
#     feat: (B, C, H, W)，希望 C=32
#     idx_b: 可视化第几个样本
#     nrow: 每行通道数
#     cmap: 灰度 → 伪彩色的 colormap（单通道时才会用到）
#     """
#     B, C, H, W = feat.shape
#     if C != 32:
#         print(f"[Warn] C={C}, not 32, but still try to visualize.")
#
#     # 取出第 idx_b 个样本
#     feat_b = feat[idx_b]          # (C, H, W)
#
#     # 统一处理成 (N,1,H,W)，N=通道数
#     feat_b = feat_b.unsqueeze(1)  # (C,1,H,W)
#
#     # 归一化到 [0,1]
#     f_min = feat_b.min()
#     f_max = feat_b.max()
#     feat_norm = (feat_b - f_min) / (f_max - f_min + 1e-8)
#
#     # 拼网格：输出 shape = (C_out, H_grid, W_grid)
#     grid = vutils.make_grid(feat_norm, nrow=nrow, padding=2)  # (C_out, H, W)
#     grid = grid.cpu()
#
#     if grid.size(0) == 1:
#         # 单通道，用 cmap 伪彩色
#         grid_np = grid.squeeze(0).numpy()        # (H, W)
#         plt.imshow(grid_np, cmap=cmap)
#     else:
#         # 多通道，当 RGB 彩色图
#         grid_np = grid.permute(1, 2, 0).numpy()  # (H, W, C_out)
#         plt.imshow(grid_np)  # 这里不要传 cmap
#
#     plt.axis("off")
#     plt.tight_layout(pad=0.01)
#
#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
#         print(f"saved to {save_path}")
#     else:
#         plt.show()

if __name__ == "__main__":
    # ======== 手动修改这几项 ========
    pth_path = "datasets/visual/mask/channel.pth"
    key = 'channel'                             # 如果 .pth 是 dict，写成 key="feat" 之类
    save_path = "channelout/feat_32_grid.png"     # 保存的可视化图
    # =================================

    feat = load_feature_from_pth(pth_path, key=key, device="cpu")  # (B,C,H,W)
    # visualize_feature_32(feat, idx_b=0, nrow=8, cmap="jet", save_path=save_path)
    visualize_feature_32(feat, idx_b=0, nrow=8, cmap_name="jet", save_path=save_path)