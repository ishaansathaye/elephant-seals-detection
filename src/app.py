import io, os, tempfile
import base64
import sys
import csv
from datetime import datetime
from flask import Flask, request, render_template_string, Response
from PIL import Image
from process import process_mosaic_in_memory, process_cropped_image  # now returns base, boxed, stats
from roboflow import Roboflow

app = Flask(__name__)

# Global stats table (in-memory) as a list of dictionaries
stats_table = []
# Global list for processed images (each entry is a base64 string)
processed_images = []

HTML_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <title>Elephant Seal Detector</title>
    
    <!-- Bootstrap CSS -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet"
    />
    <!-- Bootstrap Bundle with Popper -->
    <script
      src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"
    ></script>

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
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
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
      /* Carousel styling overrides */
      .carousel-item img {
        width: 100%;
        height: auto;
        max-height: 600px; /* adjust as needed */
        object-fit: cover;
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
      .download-btn {
        margin: 20px auto;
        display: block;
      }
      .img-box { display: none !important; }
      #processedCarousel.show-boxes .img-box { display: block !important; }
      #processedCarousel.show-boxes .img-no-box { display: none !important; }
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
      function toggleBoxes() {
        const carousel = document.getElementById('processedCarousel');
        carousel.classList.toggle('show-boxes');
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Elephant Seal Detector</h1>
      <p>Upload your image(s) to detect seals.</p>
      
      <div class="upload-section">
        <form method="post" enctype="multipart/form-data" action="/process" onsubmit="showLoading()">
          <input type="file" name="image" multiple required>
          <br/><br/>
          <!-- Toggle inside the form -->
          <div class="toggle-section">
            <label class="switch">
              <input type="checkbox" id="mosaicToggle" name="mosaic_mode" onchange="toggleChanged(this)">
              <span class="slider"></span>
            </label>
            <span id="toggleLabel">Processing as Mosaic</span>
          </div>
          <input type="submit" value="Upload" class="btn btn-primary" style="padding: 8px 16px; cursor: pointer;">
        </form>
        <p class="notice">* Larger files may take a while to process</p>
        <div id="loading-message">Processing your image(s), please wait...</div>
      </div>
      
      <div class="d-flex justify-content-center mb-3">
        <div class="form-check form-switch">
          <input class="form-check-input" type="checkbox" id="showBoxesCheckbox" onchange="toggleBoxes()">
          <label class="form-check-label" for="showBoxesCheckbox">Show Bounding Boxes</label>
        </div>
      </div>

      {% if image_data %}
        <h2>Processed Images</h2>
        <!-- Bootstrap Carousel (without auto-cycling) -->
        <div id="processedCarousel" class="carousel slide mx-auto" style="max-width: 100%;">
          <!-- Indicators -->
          <div class="carousel-indicators">
            {% for img in image_data %}
              <button type="button" data-bs-target="#processedCarousel" data-bs-slide-to="{{ loop.index0 }}"
                class="{% if loop.index0 == 0 %}active{% endif %}"
                aria-current="{% if loop.index0 == 0 %}true{% else %}false{% endif %}"
                aria-label="Slide {{ loop.index0 }}"></button>
            {% endfor %}
          </div>
          <!-- Slides -->
          <div class="carousel-inner">
            {% for img in image_data %}
              <div class="carousel-item {% if loop.index0 == 0 %}active{% endif %}">
                <img src="data:image/jpeg;base64,{{ img.no }}" class="img-no-box d-block w-100" alt="Slide {{ loop.index0 }}">
                <img src="data:image/jpeg;base64,{{ img.yes }}" class="img-box d-block w-100" alt="Slide {{ loop.index0 }}">
              </div>
            {% endfor %}
          </div>
          <!-- Controls -->
          <button class="carousel-control-prev" type="button" data-bs-target="#processedCarousel" data-bs-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Previous</span>
          </button>
          <button class="carousel-control-next" type="button" data-bs-target="#processedCarousel" data-bs-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Next</span>
          </button>
        </div>
      {% endif %}
    
    
    <div style="margin-top: 30px;"></div>
      <h3>Upload History</h3>
      <div style="margin-top: 30px; text-align: center;">
        <a href="/download" class="btn btn-success download-btn">Download Upload History as CSV</a>
      </div>
      {% if stats_table and stats_table|length > 0 %}
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Filename</th>
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
    return render_template_string(
        HTML_TEMPLATE,
        stats_table=stats_table,
        image_data=processed_images
    )

@app.route("/process", methods=["POST"])
def process():
        # Initialize Roboflow
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise Exception("ROBOFLOW_API_KEY environment variable is not set.")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("elephant-seals-project-mark-1")
    model = project.version("14").model


    if "image" not in request.files:
        return "No file part", 400
    files = request.files.getlist("image")
    if not files or files[0].filename == "":
        return "No selected file(s)", 400

    mode = "mosaic" if request.form.get("mosaic_mode") is not None else "cropped"
    print("Mode is:", mode)
    sys.stdout.flush()

    image_data_list = []
    for file in files:
        # 1. Write file.stream to a temp file on disk
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        # 2. Call your CLI‑style functions that expect a path
        if mode == "mosaic":
            # final_image, stats = process_mosaic_from_path(tmp_path)
            pass
        else:
            base_img, boxed_img, stats = process_cropped_image(tmp_path, model)

        # 3. Remove the temp file right away
        os.remove(tmp_path)

        new_stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "seals": stats.get("seals", 0),
            "male": stats.get("males", 0),
            "female": stats.get("females", 0),
            "pup": stats.get("pups", 0)
        }
        stats_table.insert(0, new_stats)

        # without boxes
        buf = io.BytesIO()
        base_img = base_img.convert("RGB")
        base_img.thumbnail((2000,2000), Image.Resampling.LANCZOS)
        base_img.save(buf, format="JPEG", quality=75)
        b64_no = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.seek(0); buf.truncate(0)
        # with boxes
        boxed_img = boxed_img.convert("RGB")
        boxed_img.thumbnail((2000,2000), Image.Resampling.LANCZOS)
        boxed_img.save(buf, format="JPEG", quality=75)
        b64_yes = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_data_list.append({"no": b64_no, "yes": b64_yes})

    global processed_images
    processed_images = image_data_list

    return render_template_string(
        HTML_TEMPLATE,
        image_data=processed_images,
        stats_table=stats_table
    )

@app.route("/download", methods=["GET"])
def download():
    # Create a CSV from stats_table
    si = io.StringIO()
    fieldnames = ["timestamp", "filename", "seals", "male", "female", "pup"]
    writer = csv.DictWriter(si, fieldnames=fieldnames)
    writer.writeheader()
    for row in stats_table:
        writer.writerow(row)
    output = si.getvalue()
    si.close()

    # Return CSV as a downloadable file
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=upload_history.csv"}
    )

if __name__ == "__main__":
    app.run(debug=True)