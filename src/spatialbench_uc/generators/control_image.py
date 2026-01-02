"""
Control image generation for ControlNet spatial guidance.

This module creates synthetic edge maps that guide ControlNet to place
objects in specific spatial arrangements (left/right, above/below).

Supports multiple rendering methods:
- rectangles: Sharp rectangles (good for GLIGEN-style bounding box models)
- ellipses: Organic ellipse shapes (better for ControlNet Canny)
- ellipses_blurred: Ellipses with Gaussian blur (best for ControlNet Canny)

Reference: PROJECT.md Section 5.4, configs/gen_sd15_controlnet.yaml
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter


# =============================================================================
# Box Coordinate Utilities
# =============================================================================

# Default placements matching gen_sd15_controlnet.yaml
DEFAULT_PLACEMENTS = {
    "left_of": {
        "box_a": {"x_range": (0.05, 0.35), "y_range": (0.25, 0.75)},
        "box_b": {"x_range": (0.65, 0.95), "y_range": (0.25, 0.75)},
    },
    "right_of": {
        "box_a": {"x_range": (0.65, 0.95), "y_range": (0.25, 0.75)},
        "box_b": {"x_range": (0.05, 0.35), "y_range": (0.25, 0.75)},
    },
    "above": {
        "box_a": {"x_range": (0.25, 0.75), "y_range": (0.05, 0.35)},
        "box_b": {"x_range": (0.25, 0.75), "y_range": (0.65, 0.95)},
    },
    "below": {
        "box_a": {"x_range": (0.25, 0.75), "y_range": (0.65, 0.95)},
        "box_b": {"x_range": (0.25, 0.75), "y_range": (0.05, 0.35)},
    },
}


def _get_box_coords(
    relation: str,
    width: int,
    height: int,
    placement_config: dict | None = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Get pixel coordinates for boxes A and B.

    Returns:
        Tuple of (box_a_coords, box_b_coords) where each is (x1, y1, x2, y2)
    """
    # Get placement for this relation
    if placement_config and relation in placement_config:
        placement = placement_config[relation]
    elif relation in DEFAULT_PLACEMENTS:
        placement = DEFAULT_PLACEMENTS[relation]
    else:
        raise ValueError(
            f"Unknown relation: {relation}. "
            f"Supported: {list(DEFAULT_PLACEMENTS.keys())}"
        )

    # Convert relative coords to pixels
    box_a = placement["box_a"]
    x1_a = int(box_a["x_range"][0] * width)
    x2_a = int(box_a["x_range"][1] * width)
    y1_a = int(box_a["y_range"][0] * height)
    y2_a = int(box_a["y_range"][1] * height)

    box_b = placement["box_b"]
    x1_b = int(box_b["x_range"][0] * width)
    x2_b = int(box_b["x_range"][1] * width)
    y1_b = int(box_b["y_range"][0] * height)
    y2_b = int(box_b["y_range"][1] * height)

    return (x1_a, y1_a, x2_a, y2_a), (x1_b, y1_b, x2_b, y2_b)


# =============================================================================
# Rendering Methods
# =============================================================================


def _draw_rectangles(
    width: int,
    height: int,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    style: dict | None = None,
) -> Image.Image:
    """
    Draw sharp rectangles (original method).

    Good for: GLIGEN-style bounding box models
    Bad for: ControlNet Canny (produces visible artifacts)
    """
    style = style or {}
    line_width = style.get("line_width", 3)
    color = style.get("color", "white")

    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    draw.rectangle(box_a, outline=color, width=line_width)
    draw.rectangle(box_b, outline=color, width=line_width)

    return img


def _draw_ellipses(
    width: int,
    height: int,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    style: dict | None = None,
) -> Image.Image:
    """
    Draw ellipses inscribed in the box regions.

    More organic than rectangles, better interpreted by ControlNet Canny.
    """
    style = style or {}
    line_width = style.get("line_width", 5)
    color = style.get("color", "white")

    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Draw ellipses inscribed in the bounding boxes
    draw.ellipse(box_a, outline=color, width=line_width)
    draw.ellipse(box_b, outline=color, width=line_width)

    return img


def _draw_ellipses_blurred(
    width: int,
    height: int,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    style: dict | None = None,
) -> Image.Image:
    """
    Draw ellipses with Gaussian blur for soft edges.

    Best for ControlNet Canny - mimics real edge detection output
    which has gradual intensity falloff rather than binary edges.
    """
    style = style or {}
    line_width = style.get("line_width", 8)
    blur_radius = style.get("blur_radius", 3)
    color = style.get("color", "white")

    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Draw thicker ellipses (they'll be softened by blur)
    draw.ellipse(box_a, outline=color, width=line_width)
    draw.ellipse(box_b, outline=color, width=line_width)

    # Apply Gaussian blur for soft edges
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def _draw_filled_ellipses_gradient(
    width: int,
    height: int,
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    style: dict | None = None,
) -> Image.Image:
    """
    Draw filled ellipses with heavy blur for gradient regions.

    Creates soft "regions of interest" rather than edge lines.
    Alternative approach if ellipse outlines still cause artifacts.
    """
    style = style or {}
    blur_radius = style.get("blur_radius", 15)
    fill_color = style.get("fill_color", (128, 128, 128))  # Gray fill

    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Draw filled ellipses
    draw.ellipse(box_a, fill=fill_color)
    draw.ellipse(box_b, fill=fill_color)

    # Heavy blur creates gradient regions
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


# =============================================================================
# Public API
# =============================================================================

# Registry of rendering methods
EDGE_MAP_METHODS = {
    "rectangles": _draw_rectangles,
    "ellipses": _draw_ellipses,
    "ellipses_blurred": _draw_ellipses_blurred,
    "gradient_regions": _draw_filled_ellipses_gradient,
}


def create_spatial_edge_map(
    relation: str,
    width: int = 512,
    height: int = 512,
    placement_config: dict | None = None,
    method: str = "rectangles",
    style: dict | None = None,
) -> Image.Image:
    """
    Create a synthetic edge map for ControlNet spatial guidance.

    Args:
        relation: Spatial relation. One of: 'left_of', 'right_of', 'above', 'below'
        width: Image width in pixels
        height: Image height in pixels
        placement_config: Optional dict with custom box placement ranges.
        method: Rendering method. One of:
            - "rectangles": Sharp rectangles (good for GLIGEN)
            - "ellipses": Organic ellipse outlines
            - "ellipses_blurred": Ellipses with soft edges (best for ControlNet)
            - "gradient_regions": Filled ellipses with heavy blur
        style: Method-specific style options:
            - line_width: Edge line thickness (default varies by method)
            - blur_radius: Gaussian blur radius for blurred methods
            - color: Edge color (default "white")

    Returns:
        PIL Image with edge map (RGB, black background)

    Example:
        ```python
        # For GLIGEN (bounding boxes)
        edge_map = create_spatial_edge_map("left_of", method="rectangles")

        # For ControlNet Canny (best quality)
        edge_map = create_spatial_edge_map("left_of", method="ellipses_blurred")
        ```
    """
    # Validate method
    if method not in EDGE_MAP_METHODS:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: {list(EDGE_MAP_METHODS.keys())}"
        )

    # Get box coordinates
    box_a, box_b = _get_box_coords(relation, width, height, placement_config)

    # Dispatch to rendering method
    render_fn = EDGE_MAP_METHODS[method]
    return render_fn(width, height, box_a, box_b, style)


def load_control_config(config: dict) -> tuple[dict | None, str, dict | None]:
    """
    Extract control image configuration from a generation config dict.

    Args:
        config: Full generation config (from YAML)

    Returns:
        Tuple of (placement_config, method, style_config)
    """
    control_config = config.get("control_image", {})

    placement = control_config.get("placement")
    method = control_config.get("method", "rectangles")
    style = control_config.get("style")

    # Handle legacy "synthetic_edges" method name
    if method == "synthetic_edges":
        method = "rectangles"

    return placement, method, style


# Keep backward compatibility
def load_placement_from_config(config: dict) -> dict | None:
    """
    Extract placement configuration from a generation config dict.

    Deprecated: Use load_control_config() instead.
    """
    control_config = config.get("control_image", {})
    return control_config.get("placement")
