# app.py
import os
import io
import base64
import uuid
from datetime import datetime

from flask import Flask, request, render_template_string
from PIL import Image

# We'll assume your mosaic_to_patches module has a function that:
#   - Takes a PIL image
#   - Runs the pipeline
#   - Returns (final_annotated_image, stats_dict)
# stats_dict might have fields like { "num_patches": X, "total_seals": Y, ... }
from mosaic_to_patches import process_mosaic_in_memory

app = Flask(__name__)

# A global list storing each upload's stats
stats_table = []

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Elephant Seal Detector</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 20px;
        background-color: #f9f9f9;
      }
      .container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      }
      h1, h2, h3 {
        text-align: center;
      }
      .upload-section {
        text-align: center;
        margin-bottom: 20px;
      }
      #loading-message {
        display: none;
        font-weight: bold;
        color: #444;
        margin-top: 10px;
        text-align: center;
      }
      .processed-image {
        display: block;
        margin: 20px auto;
        max-width: 100%;
        height: auto;
        border: 1px solid #ccc;
      }
      .notice {
        color: #666;
        font-size: 0.9em;
        text-align: center;
        margin-top: 5px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
      }
      th, td {
        padding: 8px 12px;
        border: 1px solid #ddd;
        text-align: center;
      }
      th {
        background-color: #f2f2f2;
      }
    </style>
    <script>
      function showLoading() {
        document.getElementById('loading-message').style.display = 'block';
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Elephant Seal Detector</h1>
      <p style="text-align:center; color:#555;">
        Upload your mosaic image (TIF, PNG, or JPG) to detect seals. 
      </p>
      
      <div class="upload-section">
        <form method="post" enctype="multipart/form-data" action="/process" onsubmit="showLoading()">
          <input type="file" name="image" required>
          <br/><br/>
          <input type="submit" value="Upload" style="padding: 8px 16px; cursor: pointer;">
        </form>
        <p class="notice">* Larger files may take a while to process</p>
        <div id="loading-message">
          Processing your mosaic, please wait...
        </div>
      </div>
      
      {% if image_data %}
        <h2>Processed Mosaic</h2>
        <img class="processed-image" src="data:image/jpeg;base64,{{ image_data }}" alt="Processed Mosaic">
      {% endif %}
      
      <h3 style="margin-top: 40px;">Upload History</h3>
      {% if stats_table and stats_table|length > 0 %}
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Filename</th>
            <th># Patches</th>
            <th>Total Seals</th>
            <th>Male</th>
            <th>Female</th>
            <th>Pup</th>
          </tr>
        </thead>
        <tbody>
        {% for row in stats_table %}
          <tr>
            <td>{{ row.timestamp }}</td>
            <td>{{ row.filename }}</td>
            <td>{{ row.num_patches }}</td>
            <td>{{ row.total_seals }}</td>
            <td>{{ row.male }}</td>
            <td>{{ row.female }}</td>
            <td>{{ row.pup }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
      {% else %}
        <p style="text-align:center; color:#777;">No uploads yet.</p>
      {% endif %}
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, stats_table=stats_table)

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return "No file part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    try:
        pil_image = Image.open(file.stream)
    except Exception as e:
        return f"Error opening image: {e}", 400

    # Run your mosaic pipeline entirely in memory:
    #   final_image: PIL image with bounding boxes
    #   stats: dict with e.g. {"num_patches": 50, "total_seals": 42, "male": 15, "female": 20, "pup": 7}
    final_image, stats = process_mosaic_in_memory(pil_image)

    # Build a stats record
    new_stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "num_patches": stats.get("valid_patches", 0),
        "total_seals": stats.get("total_seals", 0),
        "male": stats.get("males", 0),
        "female": stats.get("females", 0),
        "pup": stats.get("pups", 0)
    }
    stats_table.append(new_stats)

    # Convert final_image to base64
    final_image = final_image.convert("RGB")
    final_image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    final_image.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return render_template_string(
        HTML_TEMPLATE,
        image_data=img_base64,
        stats_table=stats_table
    )

if __name__ == "__main__":
    app.run(debug=True)