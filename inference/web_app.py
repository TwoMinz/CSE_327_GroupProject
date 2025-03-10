import os
import sys
import argparse
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, render_template, jsonify, send_from_directory

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.vit_model import CrowdViT, load_model_config, create_model_from_config
from inference.predict import load_model, predict_wait_time


app = Flask(__name__, template_folder="templates", static_folder="static")

# Global variables for model and config
MODEL = None
CONFIG = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_global_model(checkpoint_path, config_path=None):
    """Load model and make it globally available"""
    global MODEL, CONFIG

    if MODEL is None:
        MODEL, CONFIG = load_model(checkpoint_path, config_path)
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()

    return MODEL, CONFIG


def process_uploaded_image(file_storage, config):
    """Process uploaded image file"""
    # Read image bytes
    image_bytes = file_storage.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get target size from config
    target_size = tuple(config["data"]["image_size"])

    # Create transform
    transform = A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    return image_tensor, image


def get_base64_image(image):
    """Convert numpy image to base64 string for web display"""
    # Convert to PIL Image
    pil_img = Image.fromarray(image)

    # Save to bytes buffer
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")

    # Get base64 string
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{img_str}"


def get_wait_time_category(minutes):
    """Get wait time category based on minutes"""
    if minutes < 5:
        return "Very Short", "text-success"
    elif minutes < 15:
        return "Short", "text-info"
    elif minutes < 30:
        return "Moderate", "text-warning"
    elif minutes < 45:
        return "Long", "text-warning"
    else:
        return "Very Long", "text-danger"


@app.route("/")
def index():
    """Render main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request"""
    # Check if file was uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Check if file is empty
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        # Process image
        image_tensor, original_image = process_uploaded_image(file, CONFIG)

        # Make prediction
        results = predict_wait_time(MODEL, image_tensor, DEVICE)

        # Get wait time category
        wait_time = results.get("wait_time", 0)
        category, category_class = get_wait_time_category(wait_time)

        # Prepare results
        response = {
            "wait_time": round(wait_time, 1),
            "category": category,
            "category_class": category_class,
            "image": get_base64_image(original_image),
        }

        if "people_count" in results:
            response["people_count"] = round(results["people_count"], 1)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/static/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory("static", path)


def create_web_app_files():
    """Create necessary HTML, CSS, and JS files for the web app"""
    # Create templates directory
    os.makedirs("templates", exist_ok=True)

    # Create static directory
    os.makedirs("static", exist_ok=True)

    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>식당 대기 시간 예측 서비스</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header class="text-center my-4">
            <h1>식당 대기 시간 예측</h1>
            <p class="lead">학식 식당의 사진을 업로드하면 현재 대기 시간을 예측해드립니다.</p>
        </header>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">이미지 업로드</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-form">
                            <div class="mb-3">
                                <label for="imageInput" class="form-label">식당 사진 선택</label>
                                <input class="form-control" type="file" id="imageInput" accept="image/*">
                            </div>
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">대기 시간 예측하기</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">예측 결과</h5>
                    </div>
                    <div class="card-body">
                        <div id="loading" class="text-center d-none">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p>예측 중...</p>
                        </div>
                        
                        <div id="results" class="d-none">
                            <div class="result-item">
                                <h6>예상 대기 시간:</h6>
                                <p class="h3"><span id="wait-time">--</span> 분</p>
                            </div>
                            
                            <div class="result-item">
                                <h6>대기 시간 분류:</h6>
                                <p><span id="wait-category" class="badge">--</span></p>
                            </div>
                            
                            <div id="people-count-container" class="result-item">
                                <h6>예상 인원:</h6>
                                <p><span id="people-count">--</span> 명</p>
                            </div>
                        </div>
                        
                        <div id="error-message" class="alert alert-danger d-none">
                            오류가 발생했습니다.
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">분석된 이미지</h5>
                    </div>
                    <div class="card-body text-center">
                        <img id="preview-image" src="/static/placeholder.jpg" class="img-fluid rounded" alt="미리보기">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">대기 시간 안내</h5>
                    </div>
                    <div class="card-body">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>대기 시간</th>
                                    <th>분류</th>
                                    <th>추천 행동</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>5분 미만</td>
                                    <td><span class="badge text-bg-success">매우 짧음</span></td>
                                    <td>바로 이용 가능</td>
                                </tr>
                                <tr>
                                    <td>5-15분</td>
                                    <td><span class="badge text-bg-info">짧음</span></td>
                                    <td>잠시 기다리면 이용 가능</td>
                                </tr>
                                <tr>
                                    <td>15-30분</td>
                                    <td><span class="badge text-bg-warning">보통</span></td>
                                    <td>혼잡 시간대, 여유 있게 방문 권장</td>
                                </tr>
                                <tr>
                                    <td>30-45분</td>
                                    <td><span class="badge text-bg-warning">긺</span></td>
                                    <td>매우 혼잡한 시간대, 다른 선택 고려</td>
                                </tr>
                                <tr>
                                    <td>45분 이상</td>
                                    <td><span class="badge text-bg-danger">매우 긺</span></td>
                                    <td>다른 식당 이용 권장</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="text-center mt-4 mb-5">
            <p>CrowdViT - 학식 식당 대기 시간 예측 서비스</p>
        </footer>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Custom JS -->
    <script src="/static/script.js"></script>
</body>
</html>
    """

    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    # Create style.css
    style_css = """
body {
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.result-item {
    margin-bottom: 15px;
}

#preview-image {
    max-height: 400px;
    width: auto;
    margin: 0 auto;
}

footer {
    color: #6c757d;
}
    """

    with open("static/style.css", "w") as f:
        f.write(style_css)

    # Create script.js
    script_js = """
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('preview-image');
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    const errorElement = document.getElementById('error-message');
    const waitTimeElement = document.getElementById('wait-time');
    const waitCategoryElement = document.getElementById('wait-category');
    const peopleCountElement = document.getElementById('people-count');
    const peopleCountContainer = document.getElementById('people-count-container');
    
    // Preview image when selected
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            // Reset results
            resetResults();
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('이미지를 선택해주세요.');
            return;
        }
        
        // Show loading
        showLoading();
        
        // Create form data
        const formData = new FormData();
        formData.append('image', file);
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || '예측 중 오류가 발생했습니다.');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            hideLoading();
            
            // Show results
            showResults(data);
        })
        .catch(error => {
            // Hide loading
            hideLoading();
            
            // Show error
            showError(error.message);
        });
    });
    
    function showLoading() {
        loadingElement.classList.remove('d-none');
        resultsElement.classList.add('d-none');
        errorElement.classList.add('d-none');
    }
    
    function hideLoading() {
        loadingElement.classList.add('d-none');
    }
    
    function showResults(data) {
        // Update wait time
        waitTimeElement.textContent = data.wait_time;
        
        // Update wait category
        waitCategoryElement.textContent = data.category;
        waitCategoryElement.className = `badge bg-${data.category_class.split('-')[1]}`;
        
        // Update people count if available
        if (data.people_count !== undefined) {
            peopleCountElement.textContent = data.people_count;
            peopleCountContainer.classList.remove('d-none');
        } else {
            peopleCountContainer.classList.add('d-none');
        }
        
        // Update image if available
        if (data.image) {
            previewImage.src = data.image;
        }
        
        // Show results
        resultsElement.classList.remove('d-none');
    }
    
    function showError(message) {
        errorElement.textContent = message;
        errorElement.classList.remove('d-none');
        resultsElement.classList.add('d-none');
    }
    
    function resetResults() {
        resultsElement.classList.add('d-none');
        errorElement.classList.add('d-none');
        waitTimeElement.textContent = '--';
        waitCategoryElement.textContent = '--';
        waitCategoryElement.className = 'badge';
        peopleCountElement.textContent = '--';
    }
});
    """

    with open("static/script.js", "w") as f:
        f.write(script_js)

    # Create a placeholder image
    placeholder_img = np.ones((300, 400, 3), dtype=np.uint8) * 240
    cv2.putText(
        placeholder_img,
        "No Image",
        (150, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (100, 100, 100),
        2,
    )
    cv2.imwrite("static/placeholder.jpg", placeholder_img)


def main():
    parser = argparse.ArgumentParser(
        description="Run web application for CrowdViT model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to model configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to run the web application on",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the web application on"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Create necessary web application files"
    )
    args = parser.parse_args()

    # Create web application files if requested
    if args.setup:
        create_web_app_files()
        print("Web application files created successfully!")

    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' does not exist")
        return

    # Load model
    print("Loading model...")
    load_global_model(args.checkpoint, args.config)
    print(f"Model loaded successfully on {DEVICE}")

    # Run web application
    print(f"Starting web application on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
