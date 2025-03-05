import io
import base64
import sys
from datetime import datetime
from flask import Flask, request, render_template_string
from PIL import Image
from mosaic_to_patches import process_mosaic_in_memory, process_cropped_image

app = Flask(__name__)

# Global stats table (in-memory)
stats_table = []

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Elephant Seal Detector</title>
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f1f1f1;
      }
      .container {
        max-width: 900px;
        margin: 40px auto;
        padding: 30px;
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }
      h1, h2, h3 {
        text-align: center;
        margin-bottom: 20px;
        color: #333;
      }
      p {
        text-align: center;
        color: #555;
      }
      .upload-section {
        text-align: center;
        margin-bottom: 20px;
      }
      .toggle-section {
        text-align: center;
        margin-bottom: 20px;
      }
      /* Toggle switch styling */
      .switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 34px;
        vertical-align: middle;
      }
      .switch input {
        display: none;
      }
      .slider {
        position: absolute;
        cursor: pointer;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 34px;
      }
      .slider:before {
        position: absolute;
        content: "";
        height: 26px;
        width: 26px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
      }
      input:checked + .slider {
        background-color: #2196F3;
      }
      input:checked + .slider:before {
        transform: translateX(26px);
      }
      #toggleLabel {
        font-size: 1.1em;
        margin-left: 10px;
        vertical-align: middle;
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
        border: 1px solid #ccc;
        border-radius: 5px;
      }
      .notice {
        color: #777;
        font-size: 0.9em;
        text-align: center;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 30px;
      }
      th, td {
        padding: 10px;
        border: 1px solid #ddd;
        text-align: center;
      }
      th {
        background-color: #f7f7f7;
      }
    </style>
    <script>
      // On page load, restore toggle state from localStorage
      window.addEventListener("DOMContentLoaded", () => {
        const mode = localStorage.getItem("processingMode") || "mosaic";
        const checkbox = document.getElementById("mosaicToggle");
        checkbox.checked = (mode === "mosaic");
        updateToggleLabel(checkbox.checked);
      });
      // Update label text based on toggle state
      function updateToggleLabel(isMosaic) {
        const label = document.getElementById("toggleLabel");
        label.innerText = isMosaic ? "Processing as Mosaic" : "Processing as Cropped Image";
      }
      // When toggle is changed, update localStorage and label
      function toggleChanged(checkbox) {
        const mode = checkbox.checked ? "mosaic" : "cropped";
        localStorage.setItem("processingMode", mode);
        updateToggleLabel(checkbox.checked);
      }
      function showLoading() {
        document.getElementById('loading-message').style.display = 'block';
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Elephant Seal Detector</h1>
      <p>Upload your image to detect seals.</p>
      
      <div class="upload-section">
        <form method="post" enctype="multipart/form-data" action="/process" onsubmit="showLoading()">
          <input type="file" name="image" required>
          <br/><br/>
          <!-- Toggle placed inside the form so its value is submitted -->
          <div class="toggle-section">
            <label class="switch">
              <input type="checkbox" id="mosaicToggle" name="mosaic_mode" onchange="toggleChanged(this)">
              <span class="slider"></span>
            </label>
            <span id="toggleLabel">Processing as Mosaic</span>
          </div>
          <input type="submit" value="Upload" style="padding: 8px 16px; cursor: pointer;">
        </form>
        <p class="notice">* Larger files may take a while to process</p>
        <div id="loading-message">Processing your image, please wait...</div>
      </div>
      
      {% if image_data %}
        <h2>Processed Image</h2>
        <img class="processed-image" src="data:image/jpeg;base64,{{ image_data }}" alt="Processed Image">
      {% endif %}
      
      <h3>Upload History</h3>
      {% if stats_table and stats_table|length > 0 %}
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Filename</th>
            <th># Clumps</th>
            <th># Seals</th>
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
            <td>{{ row.clumps }}</td>
            <td>{{ row.seals }}</td>
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

    # Determine mode: if mosaic_mode checkbox is present, it means it's checked (mosaic mode).
    mode = "mosaic" if request.form.get("mosaic_mode") is not None else "cropped"
    print("Mode is:", mode)
    sys.stdout.flush()

    if mode == "mosaic":
        final_image, stats = process_mosaic_in_memory(pil_image)
    else:
        final_image, stats = process_cropped_image(pil_image)

    new_stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "clumps": stats.get("clumps", 0),
        "seals": stats.get("seals", 0),
        "male": stats.get("males", 0),
        "female": stats.get("females", 0),
        "pup": stats.get("pups", 0)
    }
    stats_table.append(new_stats)

    buf = io.BytesIO()
    final_image = final_image.convert("RGB")
    final_image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
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