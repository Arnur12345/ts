from __future__ import annotations

from pathlib import Path
from typing import Any


def resample_crop_hu(
    axial_hu: Any,
    center_row_col: tuple[float, float],
    source_spacing_mm: float,
    target_spacing_mm: float,
    size_px: int,
    outside_hu: float,
) -> Any:
    """Cubic in-plane resampling evaluated on a fixed centred crop grid."""
    import numpy as np
    from scipy.ndimage import map_coordinates

    if getattr(axial_hu, "ndim", None) != 2:
        raise ValueError("axial_hu must be a 2D array")
    center_out = (size_px - 1) / 2.0
    scale = float(target_spacing_mm) / float(source_spacing_mm)
    axis = np.arange(size_px, dtype=np.float64) - center_out
    rows = float(center_row_col[0]) + axis * scale
    cols = float(center_row_col[1]) + axis * scale
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(
        np.asarray(axial_hu, dtype=np.float32),
        [row_grid, col_grid],
        order=3,
        mode="constant",
        cval=float(outside_hu),
        prefilter=True,
    )


def window_uint8(image_hu: Any, center_hu: float, width_hu: float) -> Any:
    import numpy as np

    lower = float(center_hu) - float(width_hu) / 2.0
    upper = float(center_hu) + float(width_hu) / 2.0
    if upper <= lower:
        raise ValueError("window width must be positive")
    scaled = (np.clip(image_hu, lower, upper) - lower) * (255.0 / (upper - lower))
    return np.rint(scaled).astype(np.uint8)


def add_marker(image: Any, marker: dict[str, Any]) -> Any:
    from PIL import Image, ImageDraw

    rgb = Image.fromarray(image, mode="L").convert("RGB")
    draw = ImageDraw.Draw(rgb)
    center = (rgb.width - 1) / 2.0
    radius = int(marker["radius_px"])
    box = (center - radius, center - radius, center + radius, center + radius)
    draw.ellipse(
        box,
        outline=tuple(int(value) for value in marker["color_rgb"]),
        width=int(marker["width_px"]),
    )
    return rgb


def add_scale_bar(image: Any, spacing_mm: float, scale_bar: dict[str, Any]) -> Any:
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    margin = int(scale_bar["margin_px"])
    width = int(scale_bar["width_px"])
    length_mm = float(scale_bar["length_mm"])
    length_px = int(round(length_mm / float(spacing_mm)))
    x0, y0 = margin, result.height - margin
    x1 = x0 + length_px
    # Black under-stroke keeps the fixed white bar legible over any anatomy.
    draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=width + 4)
    draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=width)
    tick = 6
    for x in (x0, x1):
        draw.line((x, y0 - tick, x, y0 + tick), fill=(0, 0, 0), width=width + 4)
        draw.line((x, y0 - tick, x, y0 + tick), fill=(255, 255, 255), width=width)
    label = f"{length_mm:g} mm"
    font = ImageFont.load_default()
    draw.text((x0, y0 - 19), label, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return result


def save_render_arms(
    axial_hu: Any,
    center_row_col: tuple[float, float],
    source_spacing_mm: float,
    target_spacing_mm: float,
    config: dict[str, Any],
    marker_path: Path,
    scale_bar_path: Path,
) -> None:
    crop = resample_crop_hu(
        axial_hu,
        center_row_col,
        source_spacing_mm,
        target_spacing_mm,
        int(config["output_size_px"]),
        float(config["outside_hu"]),
    )
    pixels = window_uint8(crop, config["window_center_hu"], config["window_width_hu"])
    marked = add_marker(pixels, config["marker"])
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marked.save(marker_path, format="PNG", optimize=False, compress_level=9)
    barred = add_scale_bar(marked, target_spacing_mm, config["scale_bar"])
    barred.save(scale_bar_path, format="PNG", optimize=False, compress_level=9)


def make_pair(left_path: Path, right_path: Path, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    if left.size != right.size:
        raise ValueError("pair panels must have identical dimensions")
    panel_w, panel_h = left.size
    side_by_side = Image.new("RGB", (panel_w * 2, panel_h), (0, 0, 0))
    side_by_side.paste(left, (0, 0))
    side_by_side.paste(right, (panel_w, 0))
    ImageDraw.Draw(side_by_side).line((panel_w - 1, 0, panel_w - 1, panel_h), fill=(128, 128, 128), width=2)

    # Square input avoids anisotropic processor resize for fixed 896x896 models.
    side = panel_w * 2
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(side_by_side, (0, (side - panel_h) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    square.save(output_path, format="PNG", optimize=False, compress_level=9)


def make_contact_sheet(items: list[tuple[Path, str]], output_path: Path, columns: int = 4) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not items:
        raise ValueError("contact sheet requires at least one image")
    thumb = 224
    caption_h = 28
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + caption_h)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (path, label) in enumerate(items):
        with Image.open(path) as source:
            image = source.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb
        y = (index // columns) * (thumb + caption_h)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumb + 5), label, font=font, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False, compress_level=9)
