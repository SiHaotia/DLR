import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_masks(mask_pth_path, key='mask'):
    """
    从 .pth 文件加载 mask 张量，形状为 (B, N, H, W)
    如果 .pth 里是 dict，可以指定 key 取出对应的 tensor。
    """
    data = torch.load(mask_pth_path, map_location="cpu")
    if isinstance(data, dict):
        if key is None:
            raise ValueError("mask_pth 是 dict，请指定 key，例如 key='masks'。")
        masks = data[key]
    else:
        masks = data
    assert masks.ndim == 4, f"Expected masks with 4 dims (B, N, H, W), got {masks.shape}"
    return masks  # (B, N, H, W)


def load_image_paths(img_dir, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """
    从文件夹中读取所有图像路径，并按文件名排序。
    """
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
    paths = sorted(paths)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def overlay_mask_on_image(img, mask, alpha=0.5, cmap="jet"):
    """
    使用 Matplotlib 将 mask 叠加到图像上，返回 figure 对象。
    img: PIL.Image (RGB)
    mask: 2D numpy array, (H, W)，值为任意实数，将归一化到 [0, 1]
    """
    mask = mask.astype(np.float32)
    m_min, m_max = mask.min(), mask.max()
    if m_max > m_min:
        mask_norm = (mask - m_min) / (m_max - m_min + 1e-8)
    else:
        mask_norm = np.zeros_like(mask, dtype=np.float32)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask_norm, cmap=cmap, alpha=alpha)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def visualize_masks(
    mask_pth_path,
    img_dir,
    out_dir,
    mask_key=None,
    batch_indices=None,
    alpha=0.5,
):
    """
    主可视化函数：
    - mask_pth_path: .pth 文件路径，包含 (B, N, H, W) 的 mask
    - img_dir: 对应图像所在文件夹
    - out_dir: 可视化结果输出目录
    - mask_key: 若 .pth 保存为 dict，则指定取 mask 的 key；否则设为 None
    - batch_indices: 需要可视化的 batch 下标列表，None 表示全部
    - alpha: mask 叠加透明度
    """
    os.makedirs(out_dir, exist_ok=True)

    masks = load_masks(mask_pth_path, key=mask_key)  # (B, N, H, W)
    B, N, H, W = masks.shape
    print(f"Loaded masks: B={B}, N={N}, H={H}, W={W}")

    img_paths = load_image_paths(img_dir)
    if len(img_paths) < B:
        print(f"Warning: only {len(img_paths)} images found, but B={B}. "
              f"Will only visualize first {len(img_paths)} batches.")
        B = len(img_paths)

    if batch_indices is None:
        batch_indices = list(range(B))
    else:
        batch_indices = [i for i in batch_indices if 0 <= i < B]

    for b in batch_indices:
        img_path = img_paths[b]
        img = Image.open(img_path).convert("RGB")
        # 若原图大小与 mask 不一致，可以在此 resize
        img = img.resize((W, H), Image.BILINEAR)

        for n in range(N):
            mask = masks[b, n].cpu().numpy()  # (H, W)
            fig = overlay_mask_on_image(img, mask, alpha=alpha, cmap="jet")

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_name = f"b{b:02d}_cls{n:02d}_{base_name}.png"
            save_path = os.path.join(out_dir, save_name)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    # 示例用法，根据自己路径改一下
    mask_pth_path = "datasets/visual/mask/mask.pth"
    img_dir = "datasets/visual/vi"
    out_dir = "datasets/visual/out"

    # 如果 .pth 是 dict，比如 {"masks": tensor(B,N,H,W), ...}，就设 mask_key="masks"
    visualize_masks(
        mask_pth_path=mask_pth_path,
        img_dir=img_dir,
        out_dir=out_dir,
        mask_key="mask",        # or "masks"
        batch_indices=None,   # or [0, 1, 2] 只看前几个
        alpha=0.5,
    )
