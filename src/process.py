import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from collections import Counter
import pandas as pd
from joblib import load

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
