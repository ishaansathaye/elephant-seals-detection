import os
import sys
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageFont
import cv2
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
    mask = (img_np[:, :, 0] < threshold) | (img_np[:, :, 1] < threshold) | (img_np[:, :, 2] < threshold)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))

# Split the Image into Patches
def split_into_patches(image, patch_size=640):
    patches = []
    width, height = image.size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            box = (x, y, min(x + patch_size, width), min(y + patch_size, height))
            patch = image.crop(box)
            patches.append((patch, box))
    return patches

# Filter Out Mostly-White Patches
def is_patch_valid(patch, bg_color=(255, 255, 255), tolerance=5, max_fraction=0.5):
    patch_np = np.array(patch)
    close_to_bg = np.all(np.abs(patch_np - bg_color) <= tolerance, axis=-1)
    fraction_bg = np.mean(close_to_bg)
    return fraction_bg < max_fraction, fraction_bg

# Check if a patch is mostly water.
def is_patch_water(patch, water_threshold=0.3, foam_tolerance=5):
    hsv_patch = patch.convert("HSV")
    hsv_np = np.array(hsv_patch)
    lower_dark = np.array([90, 50, 50], dtype=np.uint8)
    upper_dark = np.array([140, 255, 255], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv_np, lower_dark, upper_dark)
    lower_foam = np.array([0, 0, 200], dtype=np.uint8)
    upper_foam = np.array([180, 50, 255], dtype=np.uint8)
    mask_foam = cv2.inRange(hsv_np, lower_foam, upper_foam)
    patch_np = np.array(patch)
    background_mask = np.all(np.abs(patch_np - np.array([255, 255, 255])) <= foam_tolerance, axis=-1).astype(np.uint8) * 255
    mask_foam_effective = cv2.subtract(mask_foam, background_mask)
    combined_mask = cv2.bitwise_or(mask_dark, mask_foam_effective)
    water_fraction = np.mean(combined_mask > 0)
    return water_fraction >= water_threshold, water_fraction

# Draw Bounding Boxes on Image
def draw_patch_boxes(image, patch_info, color="yellow", width=10, text_color="orange"):
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    for box in patch_info:
        box, white_frac, sand_frac = box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return drawn_image

# Visualize Patches on the Cropped Mosaic
def visualize_patches_downscaled(cropped_mosaic, patch_info, output_dir, max_dim=65500):
    w, h = cropped_mosaic.size
    if w > max_dim or h > max_dim:
        scale_factor = min(max_dim/w, max_dim/h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        mosaic_small = cropped_mosaic.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        scaled_info = []
        for (box, white_frac, sand_frac) in patch_info:
            x1, y1, x2, y2 = box
            sx1 = int(x1 * scale_factor)
            sy1 = int(y1 * scale_factor)
            sx2 = int(x2 * scale_factor)
            sy2 = int(y2 * scale_factor)
            scaled_info.append(((sx1, sy1, sx2, sy2), white_frac, sand_frac))
        mosaic_with_boxes = draw_patch_boxes(mosaic_small, scaled_info)
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(output_dir, "cropped_mosaic_with_patches_downscaled.jpg"))
        print(f"Saved downscaled visualization to cropped_mosaic_with_patches_downscaled.jpg")
    else:
        mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, patch_info)
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(output_dir, "cropped_mosaic_with_patches.jpg"))
        print(f"Saved full-resolution visualization to cropped_mosaic_with_patches.jpg")

# Main Processing Pipeline for a Mosaic (preprocessing required)
def process_mosaic_in_memory(image, patch_size=2000):
    # For JPEGs, adjust patch size if needed.
    if image.format == "JPEG":
        patch_size = 900
    print("Standardizing background...")
    sys.stdout.flush()
    mosaic = standardize_background(image)
    print("Cropping to content...")
    sys.stdout.flush()
    cropped_mosaic = crop_to_content(mosaic)
    print("Splitting into patches...")
    sys.stdout.flush()
    patches = split_into_patches(cropped_mosaic, patch_size)
    print("Filtering valid patches...")
    sys.stdout.flush()
    valid_patches = []
    patch_info = []
    for i, (patch, box) in enumerate(patches):
        valid_white, white_fraction = is_patch_valid(patch, tolerance=5, max_fraction=0.25)
        valid_water, water_fraction = is_patch_water(patch, water_threshold=0.25)
        if valid_white and not valid_water:
            valid_patches.append((patch, box))
            patch_info.append((box, white_fraction, water_fraction))
        else:
            print(f"Skipping patch {i}: white_frac={white_fraction:.2f}, water_frac={water_fraction:.2f}")
    visualize_patches_downscaled(cropped_mosaic, patch_info, output_dir="./patches")
    print("Drawing bounding boxes...")
    sys.stdout.flush()
    mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, patch_info)
    stats_dict = {
        "valid_patches": len(valid_patches),
        "total_seals": 0,
        "males": 0,
        "females": 0,
        "pups": 0
    }
    return mosaic_with_boxes, stats_dict

# Processing for a Cropped Image (no preprocessing)
def process_cropped_image(image):
    """
    This function is a placeholder for processing a cropped (small) image.
    In the future, you can add inference for the cropped image here.
    For now, it simply returns the image unchanged with dummy stats.
    """
    # Use roboflow to detect seals in the cropped image

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow( api_key=api_key)
    project = rf.workspace().project("elephant-seals-project-mark-1")
    model = project.version("14").model

    total_clumps = 0
    total_seals = 0

    # save the image to a temporary file
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # predict using the model
    result = model.predict(temp_image_path, confidence=25, overlap=30)

    output_dict = result.json().get("predictions", [])

    # Count predictions by class (clump and seals)
    clump_count = sum(1 for pred in output_dict if pred["class"] == "clump")
    seal_count = sum(1 for pred in output_dict if pred["class"] == "seals")
    total_clumps += clump_count
    total_seals += seal_count

    stats_dict = {
        "valid_patches": 0,
        "total_seals": total_seals,
        "males": 0,
        "females": 0,
        "pups": 0
    }
    print(f"Total clumps: {total_clumps}, Total seals: {total_seals}")
    sys.stdout.flush()

    # Clean up the temporary file
    os.remove(temp_image_path)
    return image, stats_dict
# import requests
# def process_cropped_image(image):
#     """
#     Process a cropped (small) image using Roboflow inference.
#     This function saves the image temporarily, sends it to the Roboflow model,
#     downloads the resulting annotated image from the returned URL,
#     and returns that image along with dummy statistics.
#     """
#     api_key = os.environ.get("ROBOFLOW_API_KEY")
#     if not api_key:
#         raise Exception("ROBOFLOW_API_KEY environment variable not set")
#     rf = Roboflow(api_key=api_key)
#     project = rf.workspace().project("elephant-seals-project-mark-1")
#     model = project.version("14").model

#     # Save the image to a temporary file
#     temp_image_path = "temp_image.jpg"
#     image.save(temp_image_path)

#     # Run prediction using Roboflow
#     result = model.predict(temp_image_path, confidence=25, overlap=30)
    
#     # Extract the URL for the annotated image from the JSON response
#     result_json = result.json()
#     print(result.plot())
#     annotated_image_url = result_json.get("image_path")
#     if not annotated_image_url:
#         os.remove(temp_image_path)
#         raise Exception("Annotated image URL not returned from Roboflow.")

#     # Download the annotated image using requests
#     response = requests.get(annotated_image_url)
#     if response.status_code != 200:
#         os.remove(temp_image_path)
#         raise Exception(f"Failed to download annotated image, status code: {response.status_code}")
    
#     final_image = Image.open(io.BytesIO(response.content))

#     # Count predictions by class
#     output_dict = result_json.get("predictions", [])
#     clump_count = sum(1 for pred in output_dict if pred["class"] == "clump")
#     seal_count = sum(1 for pred in output_dict if pred["class"] == "seals")

#     stats_dict = {
#         "valid_patches": 0,
#         "total_seals": seal_count,
#         "males": 0,
#         "females": 0,
#         "pups": 0
#     }
    
#     print(f"Total clumps: {clump_count}, Total seals: {seal_count}")
    
#     # Clean up the temporary file
#     os.remove(temp_image_path)
#     return final_image, stats_dict