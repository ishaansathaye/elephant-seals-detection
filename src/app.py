import io, os, tempfile
import base64
import sys
import csv
from datetime import datetime
from flask import Flask, request, render_template_string, Response
from PIL import Image
from process import process_cropped_image  # now returns base, boxed, stats
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
        /* Replace 'bg-gradient.jpg' with your background image path */
        background: url('/static/background.jpg') no-repeat center center fixed;
        background-size: cover;
        color: #fff;
      }
      .container {
        max-width: 1400px;
        margin: 40px auto;
        padding: 0;
        background: transparent;
      }
      h1, h2, h3 {
        text-align: center;
        margin-bottom: 20px;
        color: #fff;
      }
      p {
        text-align: center;
        color: #fff;
      }
      .upload-section {
        text-align: center;
        margin-bottom: 20px;
      }
      .main-content {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 15px;
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
        max-height: 700px; /* adjust as needed */
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
      .history-panel table {
        width: 100%;
      }
      .history-panel th {
        background-color: rgba(255, 255, 255, 0.1);
        color: #fff;
      }
      .history-panel td {
        color: #fff;
      }
      .download-btn {
        margin: 20px auto;
        display: block;
      }
      .img-box { display: none !important; }
      #processedCarousel.show-boxes .img-box { display: block !important; }
      #processedCarousel.show-boxes .img-no-box { display: none !important; }
      .glass-panel {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        flex: 1 1 48%;
        margin: 10px;
      }
      .carousel-panel .glass-panel {
        background: rgba(255,255,255,0.10) !important;
        backdrop-filter: blur(10px) !important;
      }
      .glass-panel.carousel-panel {
        flex: 0 0 50%;
        max-width: 50%;
      }

      .glass-panel.history-panel {
        flex: 0 0 45%;
        max-width: 50%;
        margin-right: 10px;
      }
      .history-panel {
        overflow-x: auto;
      }
      .history-panel table {
        width: auto;
      }
      .history-panel table {
        width: 100%;
        table-layout: fixed;
      }
      .history-panel th, .history-panel td {
        overflow-wrap: break-word;
      }
      .glass-btn {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.4);
        color: #fff;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
      }
      .glass-btn:hover {
        background: rgba(255,255,255,0.3);
      }
      /* Inner glass containers */
      .inner-glass-top {
        background: rgba(255,255,255, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
      }
      .inner-glass-bottom {
        background: rgba(255,255,255,0.05);
        display: flex;
        justify-content: center;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
      }
    </style>
    <script>
      function toggleBoxes() {
        const carousel = document.getElementById('processedCarousel');
        carousel.classList.toggle('show-boxes');
      }
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Elephant Seal Detector</h1>
      <div class="main-content d-flex justify-content-between">
        <div class="glass-panel carousel-panel p-4">
          <div class="inner-glass-top">
            <form method="post" enctype="multipart/form-data" action="/process" onsubmit="showLoading()" class="d-flex align-items-center justify-content-center gap-3">
              <label for="imageInput" style="cursor:pointer; display:flex; align-items:center; gap:10px; justify-content:center;">
                <!-- folder icon SVG -->
                <svg width="24" height="24" fill="white" viewBox="0 0 24 24">
                  <path d="M10 4H2v16h20V6H12l-2-2z"/>
                </svg>
                <span style="font-size:1.2em; color:white;">Drag &amp; drop files or click to upload</span>
              </label>
              <input id="imageInput" type="file" name="image" multiple required style="display:none;">
              <button type="submit" class="glass-btn">Upload</button>
            </form>
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
          <div class="inner-glass-bottom">
            <div class="form-check form-switch">
              <input class="form-check-input" type="checkbox" id="showBoxesCheckbox" onchange="toggleBoxes()">
              <label class="form-check-label text-white" for="showBoxesCheckbox">Show Bounding Boxes</label>
            </div>
          </div>
        </div>
        <div class="glass-panel history-panel p-4">
          <h3>Upload History</h3>
          <div style="margin-top: 30px; text-align: center;">
            <a href="/download" class="glass-btn">Download Upload History as CSV</a>
          </div>
          <div class="table-responsive">
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
              <p style="text-align:center; color:#777; margin-top:20px;">No uploads yet.</p>
            {% endif %}
          </div>
        </div>
      </div>
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

    image_data_list = []
    for file in files:
        # Write file.stream to a temp file on disk
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        base_img, boxed_img, stats = process_cropped_image(tmp_path, model)

        # Remove the temp file right away
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
        boxed_img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
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