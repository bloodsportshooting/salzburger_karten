#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# TIFF FRAME READER
# ------------------------------------------------------------

def iter_tiff_frames(path: Path):
    img = Image.open(path)
    frame = 0

    try:
        while True:
            yield frame, img.convert("RGB").copy()
            frame += 1
            img.seek(frame)
    except EOFError:
        return


# ------------------------------------------------------------
# RED BACKGROUND DETECTION
# ------------------------------------------------------------

def detect_red_background(rgb: np.ndarray) -> np.ndarray:
    """
    Detect only the red BACKGROUND.

    IMPORTANT:
    We do NOT remove red inside cards anymore.
    We only use the red mask to locate object boundaries.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 60, 40], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([168, 60, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return red_mask


# ------------------------------------------------------------
# POINT ORDERING
# ------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ------------------------------------------------------------
# CREATE CARD MASK FROM CONTOUR
# ------------------------------------------------------------

def create_precise_card_mask(shape, contour):
    """
    Create a filled contour mask.

    This preserves rounded corners naturally.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    # Smooth edges slightly
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask


# ------------------------------------------------------------
# CARD EXTRACTION
# ------------------------------------------------------------

def extract_card(
    rgb: np.ndarray,
    contour: np.ndarray,
    padding: int = 12,
):

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    center = np.array(rect[0], dtype=np.float32)

    # Expand slightly so rounded corners never clip
    expanded = []

    for p in box:
        vec = p - center
        length = np.linalg.norm(vec)

        if length == 0:
            expanded.append(p)
            continue

        scale = (length + padding) / length
        expanded.append(center + vec * scale)

    expanded = np.array(expanded, dtype=np.float32)

    src = order_points(expanded)

    width = int(
        max(
            np.linalg.norm(src[1] - src[0]),
            np.linalg.norm(src[2] - src[3]),
        )
    )

    height = int(
        max(
            np.linalg.norm(src[3] - src[0]),
            np.linalg.norm(src[2] - src[1]),
        )
    )

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    warped_rgb = cv2.warpPerspective(
        rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # IMPORTANT:
    # Build mask ONLY from outer contour.
    # Internal red graphics remain untouched.
    contour_mask = create_precise_card_mask(rgb.shape, contour)

    warped_mask = cv2.warpPerspective(
        contour_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = warped_mask.astype(np.uint8)

    rgba = np.dstack([warped_rgb, alpha])

    return rgba


# ------------------------------------------------------------
# FIND CARDS
# ------------------------------------------------------------

def detect_cards(rgb: np.ndarray, min_area: int = 20000):

    red_mask = detect_red_background(rgb)

    # Invert: non-red becomes foreground objects
    foreground = cv2.bitwise_not(red_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Filter obvious non-card shapes
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        valid.append(contour)

    return valid


# ------------------------------------------------------------
# PROCESS FRAME
# ------------------------------------------------------------

def process_frame(
    rgb: np.ndarray,
    out_dir: Path,
    base_name: str,
    frame_index: int,
):

    contours = detect_cards(rgb)

    saved = 0

    for idx, contour in enumerate(contours, start=1):

        rgba = extract_card(rgb, contour)

        out_path = (
            out_dir
            / f"{base_name}_frame_{frame_index:03d}_card_{idx:03d}.tiff"
        )

        Image.fromarray(rgba, mode="RGBA").save(
            out_path,
            format="TIFF",
            compression="tiff_lzw",
        )

        saved += 1

    return saved


# ------------------------------------------------------------
# INPUT COLLECTION
# ------------------------------------------------------------

def collect_inputs(path: Path):

    if path.is_file():
        return [path]

    files = []

    for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        files.extend(path.glob(ext))

    return sorted(files)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input TIFF or folder",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(input_path)

    if not files:
        raise SystemExit("No TIFF files found")

    total = 0

    for tif in files:

        print(f"Processing: {tif.name}")

        for frame_index, pil_img in iter_tiff_frames(tif):

            rgb = np.array(pil_img)

            count = process_frame(
                rgb,
                output_dir,
                tif.stem,
                frame_index,
            )

            total += count

            print(f"  Frame {frame_index}: {count} cards")

    print()
    print(f"Finished. Extracted {total} cards.")


if __name__ == "__main__":
    main()