import base64
import csv
import io
import os
import tempfile
from datetime import datetime

from flask import Flask, Response, render_template_string, request
from PIL import Image
from roboflow import Roboflow

from process import process_cropped_image  # now returns base, boxed, stats

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

        <!-- Google Font: Inter -->
        <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        rel="stylesheet"
        />

        <style>
            /* Panning wrapper */
            .img-wrapper {
                position: relative;
                overflow: hidden;
            }
            .img-wrapper img {
                cursor: grab;
                user-select: none;
                -webkit-user-drag: none;
                transform-origin: center center;
                will-change: transform;
            }
            .img-wrapper.grabbing img {
                cursor: grabbing;
            }
            html, body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                height: 100%;
            }
            /* Blurred background layer */
            body::before {
                content: "";
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: url('/static/background.jpg') no-repeat center center fixed;
                background-size: cover;
                filter: blur(4px);
                transform: scale(1.1);
                transform-origin: center;
                z-index: -1;
            }
            /* Ensure the main content stays sharp */
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
                margin: 0;
                padding: 0;
                background: none;
                position: relative;
                z-index: 0;
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
                color: #fff;
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
                color: #ddd;
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
                border: 1px solid #555;
                text-align: center;
            }
            th {
                background-color: rgba(70, 70, 70, 0.7);
                color: #fff;
            }
            .history-panel table {
                width: 100%;
            }
            .history-panel th {
                background-color: rgba(70, 70, 70, 0.7);
                color: #fff;
            }
            .history-panel td {
                color: #fff;
                border-color: rgba(255, 255, 255, 0.3);
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
                background: rgba(255,255,255,0.1);
                display: flex;
                justify-content: center;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }
        </style>
        <script>
            // Global pan/zoom state
            let scale = 1;
            const minScale = 0.5;
            const maxScale = 3;
            let panX = 0, panY = 0;

            function toggleBoxes() {
                const carousel = document.getElementById('processedCarousel');
                carousel.classList.toggle('show-boxes');
                updateImageScale();
            }
            function updateImageScale() {
                // Only update the currently active carousel-item's images
                const carousel = document.getElementById('processedCarousel');
                if (!carousel) return;
                const activeItem = carousel.querySelector('.carousel-item.active');
                if (!activeItem) return;
                // Both .img-no-box and .img-box
                const imgs = activeItem.querySelectorAll('.img-no-box, .img-box');
                imgs.forEach(img => {
                    img.style.transform = `translate3d(${panX}px, ${panY}px, 0) scale(${scale})`;
                    img.style.transformOrigin = 'center center';
                });
            }
            function zoomIn() {
                if (scale < maxScale) {
                    scale = Math.min(scale + 0.1, maxScale);
                    updateImageScale();
                }
            }
            function zoomOut() {
                if (scale > minScale) {
                    scale = Math.max(scale - 0.1, minScale);
                    updateImageScale();
                }
            }
            function resetView() {
                scale = 1;
                panX = 0;
                panY = 0;
                updateImageScale();
            }
            // Update zoom when carousel slides
            document.addEventListener('DOMContentLoaded', function() {
                const fileInput = document.getElementById('imageInput');
                const preview = document.getElementById('preview');
                const dropZone = document.getElementById('dropZone');
                function updatePreview() {
                    preview.innerHTML = '';
                    Array.from(fileInput.files).forEach(file => {
                        const img = document.createElement('img');
                        img.src = URL.createObjectURL(file);
                        img.style.width = '50px';
                        img.style.height = '50px';
                        img.style.objectFit = 'cover';
                        img.className = 'rounded';
                        preview.appendChild(img);
                    });
                }
                fileInput.addEventListener('change', updatePreview);
                dropZone.addEventListener('dragover', e => {
                    e.preventDefault();
                    dropZone.classList.add('border', 'border-white');
                });
                dropZone.addEventListener('dragleave', () => {
                    dropZone.classList.remove('border', 'border-white');
                });
                dropZone.addEventListener('drop', e => {
                    e.preventDefault();
                    dropZone.classList.remove('border', 'border-white');
                    fileInput.files = e.dataTransfer.files;
                    updatePreview();
                });
                // Listen for carousel slide events to reset zoom to current scale
                const carousel = document.getElementById('processedCarousel');
                if (carousel) {
                    carousel.addEventListener('slid.bs.carousel', function() {
                        updateImageScale();
                    });
                }
            });

            // Improved Panning support
            document.addEventListener('DOMContentLoaded', () => {
              const wrappers = document.querySelectorAll('.img-wrapper');
              let isPanning = false;
              let startX = 0, startY = 0;
              let originX = 0, originY = 0;
              let img = null;

              function onMouseDown(e) {
                e.preventDefault();
                isPanning = true;
                // Select the correct visible image based on bounding-boxes toggle
                const showBoxes = document.getElementById('showBoxesCheckbox').checked;
                img = this.querySelector(showBoxes ? 'img.img-box' : 'img.img-no-box');
                startX = e.clientX;
                startY = e.clientY;
                // Get current translate values from global panX/panY
                originX = panX;
                originY = panY;
                document.body.style.cursor = 'grabbing';
              }

              function onMouseMove(e) {
                if (!isPanning || !img) return;
                e.preventDefault();
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                panX = originX + dx;
                panY = originY + dy;
                img.style.transform = `translate3d(${panX}px, ${panY}px, 0) scale(${scale})`;
              }

              function onMouseUp(e) {
                if (!isPanning) return;
                isPanning = false;
                document.body.style.cursor = '';
              }

              wrappers.forEach(wrapper => {
                wrapper.addEventListener('mousedown', onMouseDown);
              });
              document.addEventListener('mousemove', onMouseMove);
              document.addEventListener('mouseup', onMouseUp);
            });
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Elephant Seal Detector</h1>
            <div class="main-content d-flex justify-content-between">
                <div class="glass-panel carousel-panel p-4">
                    <div class="inner-glass-top">
                        <form method="post" enctype="multipart/form-data" action="/process" onsubmit="showLoading()" class="d-flex align-items-center justify-content-center gap-3">
                            <label for="imageInput" id="dropZone" style="cursor:pointer; display:flex; align-items:center; gap:10px; justify-content:center;">
                                <!-- folder icon SVG -->
                                <svg width="24" height="24" fill="white" viewBox="0 0 24 24">
                                    <path d="M10 4H2v16h20V6H12l-2-2z"/>
                                </svg>
                                <span style="font-size:1.2em; color:white;">Drag &amp; drop files or click to upload</span>
                            </label>
                            <input id="imageInput" type="file" name="image" multiple required style="display:none;">
                            <button type="submit" class="glass-btn">Upload</button>
                            <div id="preview" class="d-flex gap-2 mt-2"></div>
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
                                        <div class="img-wrapper">
                                            <img src="data:image/jpeg;base64,{{ img.no }}" class="img-no-box d-block w-100" alt="Slide {{ loop.index0 }}">
                                            <img src="data:image/jpeg;base64,{{ img.yes }}" class="img-box d-block w-100" alt="Slide {{ loop.index0 }}">
                                        </div>    
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
                        <div class="d-flex align-items-center gap-3">
                            <div class="form-check form-switch mb-0">
                                <input class="form-check-input" type="checkbox" id="showBoxesCheckbox" onchange="toggleBoxes()">
                                <label class="form-check-label text-white" for="showBoxesCheckbox">Show Bounding Boxes</label>
                            </div>
                            <button class="glass-btn" id="zoomInBtn" type="button" onclick="zoomIn()" title="Zoom In">+</button>
                            <button class="glass-btn" id="zoomOutBtn" type="button" onclick="zoomOut()" title="Zoom Out">&minus;</button>
                            <button class="glass-btn" id="resetBtn" type="button" onclick="resetView()" title="Reset View">&#8634;</button>
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
                            <p style="text-align:center; color:#ddd; margin-top:20px;">No uploads yet.</p>
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
        base_img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        base_img.save(buf, format="JPEG", quality=75)
        b64_no = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.seek(0)
        buf.truncate(0)

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
