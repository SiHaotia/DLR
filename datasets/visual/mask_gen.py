import numpy as np
from PIL import Image, ImageDraw, ImageFilter

img_path = "datasets/visual/vi/img.png"
img = Image.open(img_path).convert("RGB")
w, h = img.size

# 两个不规则区域，用归一化坐标写，方便调节
poly1_norm = [
    (0.00, 0.37),
    (0.08, 0.37),
    (0.12, 0.45),
    (0.21, 0.43),
    (0.50, 1.00),
    (0.00, 1.00),
]

poly2_norm = [
    (0.60, 0.75),
    (1.0, 0.65),
    (1.00, 1.00),
    (0.70, 1.00),
]

poly1_norm = [
    (0.07, 0.00),
    (0.20, 0.00),
    (0.30, 0.30),
    (0.50, 0.35),
    (0.56, 0.70),
]

# poly1_norm = [
#     (0.65, 0.30),
#     (0.82, 0.27),
#     (0.87, 0.20),
#     (0.89, 0.20),
#     (0.97, 0.35),
#     (0.92, 0.45),
#     (0.88, 0.45),
#     (0.70, 0.38)
# ]

def norm2pixel(poly_norm, w, h):
    return [(int(x * w), int(y * h)) for x, y in poly_norm]

def jitter_points(points, noise_std=0.01):
    """对每个点加一点高斯噪声，并裁剪到 [0,1]"""
    arr = np.array(points, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape)
    arr_noisy = np.clip(arr + noise, 0.0, 1.0)
    return [tuple(p) for p in arr_noisy]

def add_noisy_points_on_edges(points, num_per_edge=2, noise_std=0.01):
    """
    在每条边上插入 num_per_edge 个点，并加一点噪声，
    让轮廓不那么规则。
    """
    pts = np.array(points, dtype=np.float32)
    new_pts = []

    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]  # 闭合多边形
        new_pts.append(tuple(p1))

        # 在边上均匀插值 num_per_edge 个点，并加入噪声
        for k in range(1, num_per_edge + 1):
            t = k / (num_per_edge + 1)
            p = (1 - t) * p1 + t * p2
            p += np.random.normal(0.0, noise_std, size=p.shape)
            p = np.clip(p, 0.0, 1.0)
            new_pts.append(tuple(p))

    return new_pts

# 先对原始点轻微抖动，再在边上加一些噪声点
poly1_norm_jitter = jitter_points(poly1_norm, noise_std=0.005)
poly1_norm_noisy = add_noisy_points_on_edges(poly1_norm_jitter,
                                             num_per_edge=30,  # 每条边多插 1 个点
                                             noise_std=0.005)

poly2_norm_jitter = jitter_points(poly2_norm, noise_std=0.005)
poly2_norm_noisy = add_noisy_points_on_edges(poly2_norm_jitter,
                                             num_per_edge=50,  # 每条边多插 1 个点
                                             noise_std=0.005)


poly1 = norm2pixel(poly1_norm_noisy, w, h)
# poly2 = norm2pixel(poly2_norm_noisy, w, h)

# 1 个 mask，包含两个区域
mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)
draw.polygon(poly1, fill=255)
# draw.polygon(poly2, fill=255)

mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))  # 半径可以试 2~4
mask_np = np.array(mask)

# 叠加到原图
overlay_np = np.array(img).copy()
red = np.array([10, 100, 255], dtype=np.uint8)
alpha = 0.5
mask_bool = (mask_np > 0)[..., None]
overlay_np = np.where(
    mask_bool,
    (alpha * red + (1 - alpha) * overlay_np).astype(np.uint8),
    overlay_np,
)
overlay_img = Image.fromarray(overlay_np)
overlay_img.save("vis_overlay_mask4.png")
# 拼接展示
# combined = Image.new("RGB", (w * 3, h))
# combined.paste(img, (0, 0))
# combined.paste(mask.convert("RGB"), (w, 0))
# combined.paste(overlay_img, (2 * w, 0))
# combined.save("vis_two_regions_mask.png")

