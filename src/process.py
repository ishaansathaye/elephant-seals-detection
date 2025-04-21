import os
import sys
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImageFont
import cv2
from pathlib import Path
import requests
from collections import Counter
import pandas as pd
from joblib import load

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
    mask = (img_np[:, :, 0] < threshold) | (img_np[:, :, 1]
                                            < threshold) | (img_np[:, :, 2] < threshold)
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
            box = (x, y, min(x + patch_size, width),
                   min(y + patch_size, height))
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
    background_mask = np.all(np.abs(
        patch_np - np.array([255, 255, 255])) <= foam_tolerance, axis=-1).astype(np.uint8) * 255
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
        mosaic_small = cropped_mosaic.resize(
            (new_w, new_h), resample=Image.Resampling.BICUBIC)
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
        mosaic_with_boxes.save(os.path.join(
            output_dir, "cropped_mosaic_with_patches_downscaled.jpg"))
        print(
            f"Saved downscaled visualization to cropped_mosaic_with_patches_downscaled.jpg")
    else:
        mosaic_with_boxes = draw_patch_boxes(cropped_mosaic, patch_info)
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(
            output_dir, "cropped_mosaic_with_patches.jpg"))
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
        valid_white, white_fraction = is_patch_valid(
            patch, tolerance=5, max_fraction=0.25)
        valid_water, water_fraction = is_patch_water(
            patch, water_threshold=0.25)
        if valid_white and not valid_water:
            valid_patches.append((patch, box))
            patch_info.append((box, white_fraction, water_fraction))
        else:
            print(
                f"Skipping patch {i}: white_frac={white_fraction:.2f}, water_frac={water_fraction:.2f}")
    visualize_patches_downscaled(
        cropped_mosaic, patch_info, output_dir="./patches")
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

# Get Individual Seals and Clumps


def get_indivs_and_clumps(model, path, seal_conf_lvl, clump_conf_lvl, overlap):
    """
    Processes a single image path and returns:
      - clump_imgs_dct: dict mapping image key to list of clump sub-images
      - ind_seals_dct: dict mapping image key to number of individual seals
      - seal_boxes_dct: dict mapping image key to list of individual seal boxes
      - clump_boxes_dct: dict mapping image key to list of clump boxes
    """
    clump_imgs_dct = {}
    ind_seals_dct = {}
    seal_boxes_dct = {}
    clump_boxes_dct = {}

    def intersects(seal, clump):
        sx1 = seal['x'] - seal['width'] / 2
        sx2 = seal['x'] + seal['width'] / 2
        sy1 = seal['y'] - seal['height'] / 2
        sy2 = seal['y'] + seal['height'] / 2
        cx1 = clump['x'] - clump['width'] / 2
        cx2 = clump['x'] + clump['width'] / 2
        cy1 = clump['y'] - clump['height'] / 2
        cy2 = clump['y'] + clump['height'] / 2
        return not (sx2 <= cx1 or sx1 >= cx2 or sy2 <= cy1 or sy1 >= cy2)

    # Load image and get predictions
    image = Image.open(path)
    preds = model.predict(
        path,
        confidence=min(seal_conf_lvl, clump_conf_lvl),
        overlap=overlap
    ).json().get('predictions', [])

    # Split out seals vs. clumps
    seals = [p for p in preds if p['class']=='seals' and p['confidence']>seal_conf_lvl/100]
    clumps = [p for p in preds if p['class']=='clump' and p['confidence']>clump_conf_lvl/100]

    # Filter out seals that lie within any clump region
    filtered_seals = [s for s in seals if not any(intersects(s, c) for c in clumps)]

    key = Path(path).stem
    ind_seals_dct[key] = len(filtered_seals)

    # Build seal‐box list
    seal_boxes = []
    for s in filtered_seals:
        x1 = s['x'] - s['width']/2
        y1 = s['y'] - s['height']/2
        x2 = s['x'] + s['width']/2
        y2 = s['y'] + s['height']/2
        seal_boxes.append((x1, y1, x2, y2))
    seal_boxes_dct[key] = seal_boxes

    # Build clump‐box list and crop subimages
    clump_boxes = []
    subimages = []
    for c in clumps:
        x1 = c['x'] - c['width']/2
        y1 = c['y'] - c['height']/2
        x2 = c['x'] + c['width']/2
        y2 = c['y'] + c['height']/2
        box = (x1, y1, x2, y2)
        clump_boxes.append(box)
        subimages.append(image.crop(box))
    clump_boxes_dct[key] = clump_boxes
    clump_imgs_dct[key] = subimages

    return clump_imgs_dct, ind_seals_dct, seal_boxes_dct, clump_boxes_dct

# Get Heuristics from Clump Images


def get_heuristics(dct):
    widths = []
    heights = []
    avg_r = []
    sd_r = []
    avg_g = []
    sd_g = []
    avg_b = []
    sd_b = [] 

    keys = []

    for key, clump_lst in dct.items():

        for idx, clump in enumerate(clump_lst): 
        
            width, height = clump.size

            widths.append(width)
            heights.append(height)

            img_array = np.array(clump)

            avg_r.append(np.mean(img_array[1, :, :]))
            sd_r.append(np.std(img_array[1, :, :]))
            avg_g.append(np.mean(img_array[:, 1, :]))
            sd_g.append(np.std(img_array[:, 1, :]))
            avg_b.append(np.mean(img_array[:, :, 1]))
            sd_b.append(np.std(img_array[:, :, 1]))

            keys.append(key)

    return pd.DataFrame({'key': keys, 'width': widths,
                        'height': heights, 'avg_r': avg_r, 
                        'sd_r': sd_r, 'avg_g': avg_g,
                        'sd_g': sd_g,'avg_b': avg_b,
                        'sd_b': sd_b})

# Processing for a Cropped Image (no preprocessing)
def process_cropped_image(image_path, model):
    """
    Processes a single cropped image using Roboflow inference and CLI logic.
    It saves the image to a temporary file, runs prediction on it, downloads the annotated image,
    and then uses the CLI functions to count clumps and individual seals and predict a final seal count.
    """

    image = Image.open(image_path).convert("RGB")
    # Keep original image
    base_image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 300)
    except IOError:
        font = ImageFont.load_default()

    # Use CLI logic to get clump and individual counts on this single image.
    # We simulate a list of one image path.
    clump_imgs_dct, ind_seals_dct, seal_boxes_dct, clump_boxes_dct = get_indivs_and_clumps(
        model, 
        image_path,
        seal_conf_lvl=20, 
        clump_conf_lvl=40, 
        overlap=20
    )
    
    clump_imgs_dct = {key: value for key, value in clump_imgs_dct.items() if len(value) >= 10}

    # Get heuristics from the clump images
    if clump_imgs_dct:
        df_heur = get_heuristics(clump_imgs_dct)
        # Load the pre-trained model to predict clump counts.
        clump_model = load('assets/random_forest_mod1.joblib')
        X = df_heur.drop(columns=['key'])

        df_heur['pred_y'] = clump_model.predict(X)
        counts = df_heur['pred_y'].tolist()

        # # print the entire df_heur DataFrame
        # print(df_heur)
        # # print column names
        # print(df_heur.columns)
        # sys.stdout.flush()
        
        clump_sums = df_heur.groupby('key')['pred_y'].sum().to_dict()
    else:
        clump_sums = {}
        counts = None

    # Combine individual seal counts with clump predictions
    final_counts = dict(Counter(ind_seals_dct) + Counter(clump_sums))
    total_seals = sum(final_counts.values())

    stats_dict = {
        # "clumps": clump_sums.get(Path(temp_image_path).stem, 0),
        "seals": round(total_seals, 2),
        "males": 0,
        "females": 0,
        "pups": 0
    }
    print(
        f"Total seals: {stats_dict['seals']}")
    sys.stdout.flush()

    # Define key for drawing boxes
    key = Path(image_path).stem
    boxes = clump_boxes_dct[key]

    # Draw boxes around individual seals in green
    for box in seal_boxes_dct.get(key, []):
        draw.rectangle(box, outline="white", width=4)
        
    # Draw clump rectangles and large labels using OpenCV
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        if counts is not None:
            # Use the count from the heuristic model
            count = counts[idx]
        else:
            count = 0
        
        # Draw rectangle (thickness 4) in BGR (yellow)
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0,255,255), thickness=4)
        
        # Choose font params for large text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 4
        
        # Determine text size
        text = str(count)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text inside the top-left corner of the box with a margin
        tx, ty = x1 + 10, y1 + th + 10
        
        # Draw black outline
        # cv2.putText(cv_img, text, (tx, ty), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        
        # Draw yellow fill
        cv2.putText(cv_img, text, (tx, ty), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)
        
        # print(f"Clump {idx+1}: {count} seals")
        # sys.stdout.flush()
    
    # Convert back to PIL Image for return
    image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    boxed_image = image
    return base_image, boxed_image, stats_dict
