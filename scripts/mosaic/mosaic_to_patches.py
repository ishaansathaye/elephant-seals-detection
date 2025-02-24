import os
import math
import numpy as np
from PIL import Image, ImageOps, ImageFile, ImageDraw
from roboflow import Roboflow

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
    mask = (img_np[:,:,0] < threshold) | (img_np[:,:,1] < threshold) | (img_np[:,:,2] < threshold)
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
            box = (x, y, min(x + patch_size, width), min(y + patch_size, height))
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

# Checks if a patch has sufficient sand content
def is_patch_sand(patch, sand_threshold=0.3):
    """
    Convert the patch to HSV and check the fraction of pixels
    that fall into the sand color range. The ranges below are
    approximate and can be tuned.
    """
    # Convert to HSV. (PIL uses 0-255 for each channel in "HSV" mode.)
    hsv_patch = patch.convert("HSV")
    patch_np = np.array(hsv_patch)
    H = patch_np[:,:,0]
    S = patch_np[:,:,1]
    V = patch_np[:,:,2]
    # Define an approximate range for sand colors:
    # - Hue between 10 and 32 (these values are approximations)
    # - Saturation at least 50 (to avoid washed-out colors)
    # - Value (brightness) at least 150
    lower_h, upper_h = 10, 32
    lower_s = 50
    lower_v = 150
    sand_mask = (H >= lower_h) & (H <= upper_h) & (S >= lower_s) & (V >= lower_v)
    fraction_sand = np.mean(sand_mask)
    return fraction_sand >= sand_threshold, fraction_sand

# Draw Bounding Boxes on Image
def draw_patch_boxes(image, patch_boxes, color="red", width=2):
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
    
    for box in patch_boxes:
        x1, y1, x2, y2 = box
        # Draw the rectangle in the chosen color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    return drawn_image

def visualize_patches_downscaled(cropped_mosaic, valid_boxes, output_dir, max_dim=65500):
    # 1. Check dimensions
    w, h = cropped_mosaic.size
    if w > max_dim or h > max_dim:
        # 2. Compute a scale factor to fit within max_dim
        scale_factor = min(max_dim / w, max_dim / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        # 3. Resize the mosaic
        mosaic_small = cropped_mosaic.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

        # 4. Scale the bounding boxes
        scaled_boxes = []
        for (x1, y1, x2, y2) in valid_boxes:
            sx1 = int(x1 * scale_factor)
            sy1 = int(y1 * scale_factor)
            sx2 = int(x2 * scale_factor)
            sy2 = int(y2 * scale_factor)
            scaled_boxes.append((sx1, sy1, sx2, sy2))

        # 5. Draw the scaled boxes
        mosaic_with_boxes = draw_patch_boxes(mosaic_small, scaled_boxes, color="yellow", width=10)

        # 6. Save
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(output_dir, "cropped_mosaic_with_patches_downscaled.jpg"))
        print(f"Saved downscaled visualization to cropped_mosaic_with_patches_downscaled.jpg")
    else:
        # If already within limits, just draw directly
        mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, valid_boxes, color="red", width=2)
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(output_dir, "cropped_mosaic_with_patches.jpg"))
        print(f"Saved full-resolution visualization to cropped_mosaic_with_patches.jpg")

# Main Processing Pipeline
def process_mosaic(image_path, patch_size=2000, output_dir="./patches"):
    # Load image
    mosaic = Image.open(image_path)
    
    # Standardize the background to ensure consistency (white background)
    mosaic = standardize_background(mosaic)
    
    # Crop the mosaic to only include the actual content (non-white areas)
    cropped_mosaic = crop_to_content(mosaic)
    
    # Optionally, save the cropped mosaic for verification:
    # os.makedirs(output_dir, exist_ok=True)
    # cropped_path = os.path.join(output_dir, "cropped_mosaic.jpg")
    # cropped_mosaic.save(cropped_path)
    
    # Split into patches using a grid
    all_patches = split_into_patches(cropped_mosaic, patch_size)
    
    # Filter valid patches
    valid_patches = []
    valid_boxes = []
    for i, (patch, box) in enumerate(all_patches):
        # TODO: Change tolerance and max_fraction
        valid_white, fraction = is_patch_valid(patch, tolerance=5, max_fraction=0.5)
        if valid_white:
            # Check if the patch has sufficient sand content
            valid_sand, sand_fraction = is_patch_sand(patch, sand_threshold=0.3)
            if valid_sand:
                valid_patches.append((patch, box))
                valid_boxes.append(box)
                # print(f"Patch {i} is valid with fraction {fraction:.2f} being close to background.")
                # print(f"Patch {i} is valid with fraction {sand_fraction:.2f} being sand.")
    
    # Visualize patch boxes on the cropped mosaic
    # Only drawing valid patch boxes, but you can draw all
    # mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, valid_boxes, color="yellow", width=10)
    # os.makedirs(output_dir, exist_ok=True)
    # mosaic_with_boxes.save(os.path.join(output_dir, "cropped_mosaic_with_patches.jpg"))
    visualize_patches_downscaled(cropped_mosaic, valid_boxes, output_dir)
    
    # Save valid patches
    for idx, (patch, box) in enumerate(valid_patches):
        patch_filename = os.path.join(output_dir, f"patch_{idx}_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg")
        patch.save(patch_filename)
    
    print(f"Total patches generated: {len(all_patches)}")
    print(f"Valid patches (with content): {len(valid_patches)}")
    return valid_patches

# Example usage:
if __name__ == "__main__":
    # image_path = "./images/DC Mosaic 2.18.23.tif"
    image_path = "./images/AL Mosaic 2.16.23.png"
    valid_patches = process_mosaic(image_path, patch_size=2500, output_dir="./patches")