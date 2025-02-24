import os
import math
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

#### Deprecated #####
# Checks if a patch has sufficient sand content
# def is_patch_sand(patch, sand_threshold=0.3):
#     """
#     Convert the patch to HSV and check the fraction of pixels
#     that fall into the sand color range. The ranges below are
#     approximate and can be tuned.
#     """
#     # Convert to HSV. (PIL uses 0-255 for each channel in "HSV" mode.)
#     hsv_patch = patch.convert("HSV")
#     patch_np = np.array(hsv_patch)
#     H = patch_np[:, :, 0]
#     S = patch_np[:, :, 1]
#     V = patch_np[:, :, 2]

#     # Adjusted thresholds (tweak as needed):
#     lower_h, upper_h = 7, 50          # Approximately 10-70° in degrees
#     lower_s, upper_s = 10, 150        # Sand usually is not very saturated
#     lower_v = 80                    # Allow for somewhat darker (wet) sand

#     sand_mask = ((H >= lower_h) & (H <= upper_h) &
#                  (S >= lower_s) & (S <= upper_s) &
#                  (V >= lower_v))

#     fraction_sand = np.mean(sand_mask)
#     return fraction_sand >= sand_threshold, fraction_sand


def is_patch_sand_kmeans(patch, sand_threshold=0.3, k=3):
    """
    Uses k-means clustering on the LAB representation of the patch to estimate
    the fraction of pixels that are sand. The approach is:
      1. Convert the patch (PIL Image) to an OpenCV image (BGR) then to LAB.
      2. Run k-means clustering on the pixel values.
      3. Evaluate each cluster's average L (lightness) value.
          We assume that sand typically has a moderate lightness (e.g., between 100 and 180).
      4. Pick the cluster that falls within this L range (if any) with the highest proportion.
      5. Return True if the fraction of pixels in that cluster is at least sand_threshold.
    Returns:
      (is_sand, sand_fraction)
    """
    # Convert PIL patch to OpenCV (RGB -> BGR)
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    # Convert to LAB color space
    patch_lab = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2LAB)
    # Reshape to a list of pixels
    pixel_values = patch_lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and run k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    # Compute the proportion of each cluster
    proportions = [np.mean(labels == i) for i in range(k)]
    # Convert centers to uint8 for easier interpretation
    centers = np.uint8(centers)
    # Extract the L (lightness) values of each cluster center
    L_values = centers[:, 0]

    # Assume sand typically has moderate lightness, e.g., between 100 and 180.
    # Find candidate clusters whose L falls in this range.
    candidate_indices = [i for i in range(k) if 100 <= L_values[i] <= 180]
    if candidate_indices:
        # Choose the candidate cluster with the highest proportion
        sand_cluster = max(candidate_indices, key=lambda i: proportions[i])
    else:
        # Fallback: choose the cluster with the highest proportion overall.
        sand_cluster = np.argmax(proportions)

    sand_fraction = np.mean(labels == sand_cluster)
    return sand_fraction >= sand_threshold, sand_fraction

import cv2
import numpy as np
import torch

def is_patch_sand_kmeans_torch(patch, sand_threshold=0.3, k=3, max_iter=100):
    """
    Uses PyTorch (with MPS if available) to perform k-means clustering on the LAB representation
    of the patch and estimate the fraction of pixels that are sand.
    
    Steps:
      1. Convert the PIL patch to an OpenCV image (BGR) and then to LAB.
      2. Flatten the LAB image to a tensor of shape (N, 3) and move it to the available device.
      3. Initialize centroids randomly from the data.
      4. Iteratively assign pixels to clusters and update centroids.
      5. Identify a candidate "sand" cluster (using the L channel range [100, 180]).
      6. Return True if the fraction of pixels assigned to that cluster is >= sand_threshold.
    
    Returns:
      (is_sand, sand_fraction)
    """
    # Convert patch (PIL Image) to OpenCV image (BGR) then to LAB
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    patch_lab = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2LAB)
    H, W, C = patch_lab.shape
    # Flatten pixels: shape (N, 3)
    pixel_values = patch_lab.reshape(-1, 3).astype(np.float32)
    
    # Use PyTorch MPS if available, otherwise CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # Convert data to tensor on the chosen device
    data = torch.tensor(pixel_values, dtype=torch.float32, device=device)
    
    # Initialize centroids: randomly select k data points
    num_pixels = data.shape[0]
    indices = torch.randperm(num_pixels)[:k]
    centroids = data[indices].clone()  # shape (k, 3)
    
    # Run iterative k-means clustering
    for iteration in range(max_iter):
        # Compute distances between each pixel and each centroid; shape: (N, k)
        distances = torch.cdist(data, centroids, p=2)
        # Assign each pixel to the closest centroid
        labels = torch.argmin(distances, dim=1)
        # Update centroids: compute mean of points assigned to each cluster
        new_centroids = []
        for j in range(k):
            mask = (labels == j)
            if torch.sum(mask) > 0:
                new_centroid = data[mask].mean(dim=0)
            else:
                # If a cluster is empty, keep the previous centroid
                new_centroid = centroids[j]
            new_centroids.append(new_centroid)
        new_centroids = torch.stack(new_centroids)
        # Check convergence (if centroids change little, break)
        if torch.norm(new_centroids - centroids) < 1e-4:
            break
        centroids = new_centroids
    
    # Compute the proportion of pixels in each cluster
    proportions = [(labels == j).float().mean().item() for j in range(k)]
    
    # Move centroids to CPU and convert to numpy for easier interpretation (as uint8)
    centers_cpu = centroids.cpu().numpy().astype(np.uint8)
    # Use the L channel (first channel in LAB) for candidate selection
    L_values = centers_cpu[:, 0]
    
    # Select candidate clusters whose L value is in the sand range (e.g., [100, 180])
    candidate_indices = [i for i in range(k) if 100 <= L_values[i] <= 180]
    if candidate_indices:
        # Among candidates, pick the cluster with the highest proportion
        sand_cluster = max(candidate_indices, key=lambda i: proportions[i])
    else:
        # Fallback: choose the cluster with the highest proportion overall
        sand_cluster = int(np.argmax(proportions))
    
    # Calculate the fraction of pixels in the chosen "sand" cluster
    sand_fraction = (labels == sand_cluster).float().mean().item()
    return sand_fraction >= sand_threshold, sand_fraction

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
    font = ImageFont.truetype("./roboto.ttf", 400)

    for box in patch_info:
        box, white_frac, sand_frac = box
        x1, y1, x2, y2 = box
        # Draw the rectangle in the chosen color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        # Create the text to display
        text = f"W:{white_frac:.2f}  S:{sand_frac:.2f}"
        # Draw the text at the top-left corner of the patch box
        draw.text((x1, y1), text, fill=text_color, font=font)
    return drawn_image


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
            cropped_mosaic, patch_info, color="yellow", width=10, text_color="red")
        os.makedirs(output_dir, exist_ok=True)
        mosaic_with_boxes.save(os.path.join(
            output_dir, "cropped_mosaic_with_patches.jpg"))
        print(f"Saved full-resolution visualization to cropped_mosaic_with_patches.jpg")

# Main Processing Pipeline


def process_mosaic(image_path, patch_size=2000, output_dir="./patches"):
    # Load image
    mosaic = Image.open(image_path)

    # Standardize the background to ensure consistency (white background)
    mosaic = standardize_background(mosaic)

    # Crop the mosaic to only include the actual content (non-white areas)
    cropped_mosaic = crop_to_content(mosaic)

    # Split into patches using a grid
    all_patches = split_into_patches(cropped_mosaic, patch_size)

    # Filter valid patches
    valid_patches = []
    patch_info = []
    for i, (patch, box) in enumerate(all_patches):
        # TODO: Change tolerance and max_fraction
        valid_white, white_fraction = is_patch_valid(
            patch, tolerance=5, max_fraction=0.5)
        if valid_white:
            # Check if the patch has sufficient sand content
            # valid_sand, sand_fraction = is_patch_sand(
            #     patch, sand_threshold=0.3)
            valid_sand, sand_fraction = is_patch_sand_kmeans(
                patch, sand_threshold=0.3, k=3)
            print(
                f"Patch {i} is valid with fraction {sand_fraction:.2f} being sand.")
            # if valid_sand:
            valid_patches.append((patch, box))
            patch_info.append((box, white_fraction, sand_fraction))
            # print(f"Patch {i} is valid with fraction {fraction:.2f} being close to background.")

    # Visualize patch boxes on the cropped mosaic
    visualize_patches_downscaled(cropped_mosaic, patch_info, output_dir)

    # Save valid patches
    # for idx, (patch, box) in enumerate(valid_patches):
    #     patch_filename = os.path.join(
    #         output_dir, f"patch_{idx}_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg")
    #     patch.save(patch_filename)

    print(f"Total patches generated: {len(all_patches)}")
    print(f"Valid patches (with content): {len(valid_patches)}")
    return valid_patches


# Example usage:
if __name__ == "__main__":
    image_path = "./images/DC Mosaic 2.18.23.tif"
    # image_path = "./images/AL Mosaic 2.16.23.png"
    valid_patches = process_mosaic(
        image_path, patch_size=2500, output_dir="./patches")
