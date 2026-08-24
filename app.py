from flask import (
    Flask,
    render_template,
    request,
    url_for,
    Response
)

from pathlib import Path
from werkzeug.utils import secure_filename
from uuid import uuid4

import os
import cv2
import numpy as np

from srd_infer import run_srd, run_srd_frame


# =====================================================
# APP SETUP
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


# =====================================================
# FOLDERS
# =====================================================

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# CONFIGURATION
# =====================================================

# Maximum uploaded image size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def allowed_file(filename: str) -> bool:
    """
    Check whether uploaded file is an allowed image type.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =====================================================
# HOME PAGE
# IMAGE UPLOAD
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    output_url = None
    error = None

    if request.method == "POST":

        # -------------------------------------------------
        # Check whether file exists
        # -------------------------------------------------

        if "image" not in request.files:

            error = "No image was uploaded."

            return render_template(
                "index.html",
                result=result,
                output_url=output_url,
                error=error
            )


        image = request.files["image"]


        # -------------------------------------------------
        # Check filename
        # -------------------------------------------------

        if image.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                result=result,
                output_url=output_url,
                error=error
            )


        # -------------------------------------------------
        # Validate extension
        # -------------------------------------------------

        if not allowed_file(image.filename):

            error = (
                "Unsupported file type. "
                "Please upload JPG, JPEG, PNG, or WEBP."
            )

            return render_template(
                "index.html",
                result=result,
                output_url=output_url,
                error=error
            )


        try:

            # -------------------------------------------------
            # Secure filename
            # -------------------------------------------------

            original_filename = secure_filename(
                image.filename
            )


            extension = Path(
                original_filename
            ).suffix.lower()


            # -------------------------------------------------
            # Generate unique ID
            # -------------------------------------------------

            unique_id = uuid4().hex


            input_filename = (
                f"{unique_id}{extension}"
            )


            output_filename = (
                f"result_{unique_id}{extension}"
            )


            input_path = (
                UPLOAD_DIR / input_filename
            )


            output_path = (
                RESULT_DIR / output_filename
            )


            # -------------------------------------------------
            # Save uploaded image
            # -------------------------------------------------

            image.save(
                str(input_path)
            )


            print(
                f"Uploaded image: {input_path}"
            )


            # -------------------------------------------------
            # YOLO / PyTorch SRD inference
            # -------------------------------------------------

            data = run_srd(
                str(input_path),
                str(output_path)
            )


            # -------------------------------------------------
            # Detection result
            # -------------------------------------------------

            result = data.get(
                "status_text",
                "Detection completed."
            )


            output_url = url_for(
                "static",
                filename=f"results/{output_filename}"
            )


        except Exception as e:

            print(
                f"[ERROR] SRD inference failed: {e}"
            )

            error = (
                "An error occurred while processing "
                "the image."
            )


    return render_template(
        "index.html",
        result=result,
        output_url=output_url,
        error=error
    )


# =====================================================
# BROWSER WEBCAM FRAME DETECTION
# =====================================================

@app.route("/detect_frame", methods=["POST"])
def detect_frame():

    try:
        # ---------------------------------------------
        # Check frame exists
        # ---------------------------------------------
        if "frame" not in request.files:
            return {
                "error": "No webcam frame received."
            }, 400

        frame_file = request.files["frame"]

        # ---------------------------------------------
        # Convert uploaded JPEG -> OpenCV image
        # ---------------------------------------------
        frame_bytes = frame_file.read()

        np_array = np.frombuffer(
            frame_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "error": "Unable to decode webcam frame."
            }, 400

        # ---------------------------------------------
        # Run SRD inference
        # ---------------------------------------------
        processed_frame, detection_data = (
            run_srd_frame(frame)
        )

        # ---------------------------------------------
        # Convert processed OpenCV image -> JPEG
        # ---------------------------------------------
        success, buffer = cv2.imencode(
            ".jpg",
            processed_frame
        )

        if not success:
            return {
                "error": "Unable to encode detection result."
            }, 500

        # ---------------------------------------------
        # Send processed image back to browser
        # ---------------------------------------------
        response = Response(
            buffer.tobytes(),
            mimetype="image/jpeg"
        )

        # Detection information in response headers
        response.headers["X-SRD-Status"] = (
            detection_data["status_text"]
        )

        response.headers["X-SRD-Compliant"] = str(
            detection_data["compliant"]
        ).lower()

        response.headers["X-SRD-Collar"] = str(
            detection_data["has_collar"]
        ).lower()

        response.headers["X-SRD-Lanyard"] = str(
            detection_data["has_lanyard"]
        ).lower()

        response.headers["X-SRD-Shoes"] = str(
            detection_data["has_shoes"]
        ).lower()

        response.headers["Cache-Control"] = "no-store"

        return response

    except Exception as e:

        print(
            f"[ERROR] Webcam frame detection failed: {e}"
        )

        return {
            "error": "Webcam detection failed."
        }, 500


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "service": "SRD Detection System"
    }, 200


# =====================================================
# FILE TOO LARGE ERROR
# =====================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        result=None,
        output_url=None,
        error=(
            "Image is too large. "
            "Maximum size is 10 MB."
        )
    ), 413


# =====================================================
# 404 ERROR
# =====================================================

@app.errorhandler(404)
def page_not_found(error):

    return {
        "error": "Page not found"
    }, 404


# =====================================================
# LOCAL DEVELOPMENT
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )


    print(
        "\n================================"
    )

    print(
        "SRD Detection System"
    )

    print(
        "================================"
    )

    print(
        f"Website:"
        f" http://localhost:{port}"
    )

    print(
        f"Webcam:"
        f" http://localhost:{port}/video_feed"
    )

    print(
        f"Health:"
        f" http://localhost:{port}/health"
    )

    print(
        "================================\n"
    )


    app.run(
        host="0.0.0.0",
        port=port,

        # You can change this to True
        # while developing.
        debug=False,

        # Useful for MJPEG streaming
        threaded=True,

        # Prevent webcam being opened twice
        use_reloader=False
    )