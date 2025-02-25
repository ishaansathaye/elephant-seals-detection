import os
import sys
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageFont
# from roboflow import Roboflow
import cv2

# Allow PIL to open very large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Standardize the Background
def standardize_background(image, bg_color=(255, 255, 255)):
    """
    If the image has an alpha channel, composite it onto a white background.
    Otherwise, convert to RGB.
    """
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=image.split()[-1])
        return background
    else:
        return image.convert("RGB")

# Crop to the Mosaic Content
def crop_to_content(image, threshold=240):
    """
    Crops out the white margins. Pixels with all channels above the threshold 
    are considered background.
    """
    img_np = np.array(image)
    # Create mask: True if any channel is less than threshold (i.e. non-white)
    mask = (img_np[:, :, 0] < threshold) | (img_np[:, :, 1]
                                            < threshold) | (img_np[:, :, 2] < threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image  # No content detected; return the original image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 to include the last pixel row/column
    return image.crop((x0, y0, x1, y1))

# Split the Image into Patches
def split_into_patches(image, patch_size=640):
    """
    Splits the image into patches on a grid.
    If the patch at the right/bottom edge is smaller than patch_size,
    it will remain at its natural size.
    Returns a list of tuples: (patch, (x, y, width, height)).
    """
    patches = []
    width, height = image.size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Ensure the box does not extend past the image dimensions
            box = (x, y, min(x + patch_size, width),
                   min(y + patch_size, height))
            patch = image.crop(box)
            patches.append((patch, box))
    return patches

# Filter Out Mostly-White Patches
def is_patch_valid(patch, bg_color=(255, 255, 255), tolerance=5, max_fraction=0.5):
    """
    Returns True if less than max_fraction of the patch's pixels are within 
    tolerance of the background color.
    """
    patch_np = np.array(patch)
    # Boolean mask: True where pixel is close to background color
    close_to_bg = np.all(np.abs(patch_np - bg_color) <= tolerance, axis=-1)
    fraction_bg = np.mean(close_to_bg)
    return fraction_bg < max_fraction, fraction_bg

# Check if a patch is mostly water.
def is_patch_water(patch, water_threshold=0.3, foam_tolerance=5):
    """
    Convert the patch to HSV and compute water fraction using two masks:
      - A dark water mask (for blue water)
      - A white foam mask, from which we subtract a strict white background mask.
    Returns (True, water_fraction) if the combined water fraction is >= water_threshold.
    """
    hsv_patch = patch.convert("HSV")
    hsv_np = np.array(hsv_patch)

    # Mask for dark blue water:
    lower_dark = np.array([90, 50, 50], dtype=np.uint8)
    upper_dark = np.array([140, 255, 255], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv_np, lower_dark, upper_dark)

    # Mask for white foam (potential water foam):
    lower_foam = np.array([0, 0, 200], dtype=np.uint8)
    upper_foam = np.array([180, 50, 255], dtype=np.uint8)
    mask_foam = cv2.inRange(hsv_np, lower_foam, upper_foam)

    # Create a strict white background mask from the RGB patch.
    patch_np = np.array(patch)
    background_mask = np.all(np.abs(
        patch_np - np.array([255, 255, 255])) <= foam_tolerance, axis=-1).astype(np.uint8) * 255

    # Subtract the pure white background from the foam mask:
    mask_foam_effective = cv2.subtract(mask_foam, background_mask)

    # Combine dark water and effective foam masks:
    combined_mask = cv2.bitwise_or(mask_dark, mask_foam_effective)

    # Compute water fraction:
    water_fraction = np.mean(combined_mask > 0)
    return water_fraction >= water_threshold, water_fraction

# Draw Bounding Boxes on Image
def draw_patch_boxes(image, patch_info, color="yellow", width=10, text_color="orange"):
    """
    image: PIL Image object
    patch_boxes: list of bounding boxes (x1, y1, x2, y2)
    color: outline color for the boxes
    width: thickness of the rectangle outline
    Returns a copy of `image` with rectangles drawn on it.
    """
    # Make a copy so you don't modify the original in memory
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    # font = ImageFont.truetype("./roboto.ttf", 350)

    for box in patch_info:
        box, white_frac, sand_frac = box
        x1, y1, x2, y2 = box
        # Draw the rectangle in the chosen color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        # Create the text to display
        # text = f"W:{white_frac:.2f}  W:{sand_frac:.2f}"
        # Draw the text at the top-left corner of the patch box
        # draw.text((x1, y1), text, fill=text_color, font=font)
    return drawn_image

# Visualize Patches on the Cropped Mosaic
def visualize_patches_downscaled(cropped_mosaic, patch_info, output_dir, max_dim=65500):
    # Check dimensions
    w, h = cropped_mosaic.size
    if w > max_dim or h > max_dim:
        # Compute a scale factor to fit within max_dim
        scale_factor = min(max_dim / w, max_dim / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        # Resize the mosaic
        mosaic_small = cropped_mosaic.resize(
            (new_w, new_h), resample=Image.Resampling.BICUBIC)

        # Scale the bounding boxes
        scaled_info = []
        for (box, white_frac, sand_frac) in patch_info:
            x1, y1, x2, y2 = box
            sx1 = int(x1 * scale_factor)
            sy1 = int(y1 * scale_factor)
            sx2 = int(x2 * scale_factor)
            sy2 = int(y2 * scale_factor)
            scaled_info.append(((sx1, sy1, sx2, sy2), white_frac, sand_frac))

        # Draw the scaled boxes
        mosaic_with_boxes = draw_patch_boxes(
            mosaic_small, scaled_info, color="yellow", width=10, text_color="orange")

        # Save
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(
            output_dir, "cropped_mosaic_with_patches_downscaled.jpg"))
        print(
            f"Saved downscaled visualization to cropped_mosaic_with_patches_downscaled.jpg")
    else:
        # If already within limits, just draw directly
        mosaic_with_boxes = draw_patch_boxes(
            cropped_mosaic, patch_info, color="yellow", width=10, text_color="orange")
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(
            output_dir, "cropped_mosaic_with_patches.jpg"))
        print(f"Saved full-resolution visualization to cropped_mosaic_with_patches.jpg")

# Main Processing Pipeline
def process_mosaic_in_memory(image_path, patch_size=2000):
    # Standardize the background to ensure consistency (white background)
    print("Standardizing background...")
    sys.stdout.flush()
    mosaic = standardize_background(mosaic)

    # Crop the mosaic to only include the actual content (non-white areas)
    print("Cropping to content...")
    sys.stdout.flush()
    cropped_mosaic = crop_to_content(mosaic)

    # Split into patches using a grid
    print("Splitting into patches...")
    sys.stdout.flush()
    patches = split_into_patches(cropped_mosaic, patch_size)

    # Filter valid patches
    print("Filtering valid patches...")
    sys.stdout.flush()
    valid_patches = []
    patch_info = []
    for i, (patch, box) in enumerate(patches):
        valid_white, white_fraction = is_patch_valid(
            patch, tolerance=5, max_fraction=0.25)
        # Check if the patch is mostly water
        valid_water, water_fraction = is_patch_water(
            patch, water_threshold=0.25)
        if valid_white and not valid_water:
            valid_patches.append((patch, box))
            patch_info.append((box, white_fraction, water_fraction))

    # Visualize patch boxes on the cropped mosaic
    # visualize_patches_downscaled(cropped_mosaic, patch_info, output_dir)

    # Draw bounding boxes on the mosaic in memory
    print("Drawing bounding boxes...")
    sys.stdout.flush()
    mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, patch_info)

    # Save valid patches
    # for idx, (patch, box) in enumerate(valid_patches):
    #     patch_filename = os.path.join(
    #         output_dir, f"patch_{idx}_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg")
    #     patch.save(patch_filename)

    # Build Stats
    stats_dict = {
        "valid_patches": len(valid_patches),
        "total_seals": 0,
        "males": 0,
        "females": 0,
        "pups": 0
    }

    # print(f"Total patches generated: {len(patches)}")
    # print(f"Valid patches (with content): {len(valid_patches)}")
    return mosaic_with_boxes, stats_dict


# Example usage:
# if __name__ == "__main__":
    # image_path = "./images/DC Mosaic 2.18.23.tif"
    # image_path = "./images/AL Mosaic 2.16.23.png"
    # valid_patches = process_mosaic(
    #     image_path, patch_size=2500, output_dir="./patches")
