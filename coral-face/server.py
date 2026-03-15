"""
Coral Face Recognition Server
DeepStack-compatible REST API for Double-Take integration.

Face detection: Coral USB TPU (MobileNet SSD v2 Face)
Face embedding: TFLite CPU (MobileFaceNet)
"""

import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coral-face")

app = Flask(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
FACES_DB = DATA_DIR / "faces.json"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/opt/coral-face/models"))

DETECT_MODEL = MODELS_DIR / "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
EMBED_MODEL = MODELS_DIR / "mobilefacenet.tflite"

DETECT_THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.5"))
RECOGNIZE_THRESHOLD = float(os.environ.get("RECOGNIZE_THRESHOLD", "0.45"))

_detect_interpreter = None
_embed_interpreter = None
_faces_db = {}  # {name: [embedding_array, ...]}


def load_faces_db():
    """Load face database from JSON index + numpy files."""
    global _faces_db
    _faces_db = {}
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not FACES_DB.exists():
        return

    with open(FACES_DB, "r") as f:
        index = json.load(f)

    for name, filenames in index.items():
        embeddings = []
        for fname in filenames:
            path = EMBEDDINGS_DIR / fname
            if path.exists():
                embeddings.append(np.load(str(path)))
        if embeddings:
            _faces_db[name] = embeddings

    log.info(f"Loaded {len(_faces_db)} faces from database")


def save_faces_db():
    """Save face database as JSON index + numpy files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    index = {}
    for name, embeddings in _faces_db.items():
        filenames = []
        for i, emb in enumerate(embeddings):
            fname = f"{name}_{i}.npy"
            np.save(str(EMBEDDINGS_DIR / fname), emb)
            filenames.append(fname)
        index[name] = filenames

    with open(FACES_DB, "w") as f:
        json.dump(index, f)


def init_detect_interpreter():
    """Initialize face detection interpreter on Coral TPU."""
    global _detect_interpreter
    try:
        from pycoral.utils.edgetpu import make_interpreter
        _detect_interpreter = make_interpreter(str(DETECT_MODEL))
        _detect_interpreter.allocate_tensors()
        log.info("Face detection model loaded on Coral TPU")
        return True
    except Exception as e:
        log.warning(f"Coral TPU not available, falling back to CPU: {e}")
        try:
            import tflite_runtime.interpreter as tflite
            cpu_model = MODELS_DIR / "ssd_mobilenet_v2_face_quant_postprocess.tflite"
            model_path = str(cpu_model) if cpu_model.exists() else str(DETECT_MODEL)
            _detect_interpreter = tflite.Interpreter(model_path=model_path)
            _detect_interpreter.allocate_tensors()
            log.info("Face detection model loaded on CPU (fallback)")
            return True
        except Exception as e2:
            log.error(f"Failed to load detection model: {e2}")
            return False


def init_embed_interpreter():
    """Initialize face embedding interpreter (CPU TFLite)."""
    global _embed_interpreter
    try:
        import tflite_runtime.interpreter as tflite
        _embed_interpreter = tflite.Interpreter(model_path=str(EMBED_MODEL))
        _embed_interpreter.allocate_tensors()
        log.info("Face embedding model loaded on CPU")
        return True
    except Exception as e:
        log.error(f"Failed to load embedding model: {e}")
        return False


def detect_faces(image: Image.Image) -> list:
    """Detect faces using Coral TPU. Returns list of {box, confidence}."""
    if _detect_interpreter is None:
        return []

    input_details = _detect_interpreter.get_input_details()
    output_details = _detect_interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    height, width = input_shape[1], input_shape[2]

    img_resized = image.resize((width, height), Image.LANCZOS)
    input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)

    _detect_interpreter.set_tensor(input_details[0]["index"], input_data)
    _detect_interpreter.invoke()

    boxes = _detect_interpreter.get_tensor(output_details[0]["index"])[0]
    scores = _detect_interpreter.get_tensor(output_details[2]["index"])[0]

    img_w, img_h = image.size
    faces = []
    for i in range(len(scores)):
        if scores[i] >= DETECT_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            faces.append({
                "box": {
                    "x_min": max(0, int(xmin * img_w)),
                    "y_min": max(0, int(ymin * img_h)),
                    "x_max": min(img_w, int(xmax * img_w)),
                    "y_max": min(img_h, int(ymax * img_h)),
                },
                "confidence": float(scores[i]),
            })
    return faces


def get_face_embedding(image: Image.Image, box: dict):
    """Extract face embedding using MobileFaceNet TFLite model."""
    if _embed_interpreter is None:
        return None

    x1, y1 = box["x_min"], box["y_min"]
    x2, y2 = box["x_max"], box["y_max"]

    w, h = x2 - x1, y2 - y1
    margin = int(max(w, h) * 0.2)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.width, x2 + margin)
    y2 = min(image.height, y2 + margin)

    face_crop = image.crop((x1, y1, x2, y2))

    input_details = _embed_interpreter.get_input_details()
    output_details = _embed_interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    face_h, face_w = input_shape[1], input_shape[2]

    face_resized = face_crop.resize((face_w, face_h), Image.LANCZOS)
    face_array = np.array(face_resized, dtype=np.float32)

    if input_details[0]["dtype"] == np.uint8:
        face_array = np.array(face_resized, dtype=np.uint8)
    else:
        face_array = (face_array - 127.5) / 128.0

    input_data = np.expand_dims(face_array, axis=0)
    _embed_interpreter.set_tensor(input_details[0]["index"], input_data)
    _embed_interpreter.invoke()

    embedding = _embed_interpreter.get_tensor(output_details[0]["index"])[0].copy()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def find_match(embedding):
    """Find best matching face in database."""
    best_name = "unknown"
    best_score = 0.0

    for name, embeddings in _faces_db.items():
        for stored_emb in embeddings:
            score = cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score < RECOGNIZE_THRESHOLD:
        return "unknown", best_score

    return best_name, best_score


def read_image_from_request():
    for key in ["image", "image1", "file"]:
        if key in request.files:
            return Image.open(request.files[key].stream).convert("RGB")
    return None


# --- DeepStack-compatible API ---

@app.route("/v1/vision/face/register", methods=["POST"])
def register_face():
    userid = request.form.get("userid")
    if not userid:
        return jsonify({"success": False, "error": "userid required"}), 400

    image = read_image_from_request()
    if image is None:
        return jsonify({"success": False, "error": "image required"}), 400

    start = time.time()
    faces = detect_faces(image)
    if not faces:
        return jsonify({"success": False, "error": "no face detected", "processMs": 0})

    largest = max(faces, key=lambda f: (f["box"]["x_max"] - f["box"]["x_min"]) * (f["box"]["y_max"] - f["box"]["y_min"]))
    embedding = get_face_embedding(image, largest["box"])
    if embedding is None:
        return jsonify({"success": False, "error": "embedding failed"}), 500

    if userid not in _faces_db:
        _faces_db[userid] = []
    _faces_db[userid].append(embedding)
    save_faces_db()

    elapsed = int((time.time() - start) * 1000)
    log.info(f"Registered face for '{userid}' ({len(_faces_db[userid])} samples)")
    return jsonify({"success": True, "message": f"face registered for {userid}", "processMs": elapsed})


@app.route("/v1/vision/face/recognize", methods=["POST"])
def recognize_face():
    start = time.time()

    image = read_image_from_request()
    if image is None:
        return jsonify({"success": False, "error": "image required"}), 400

    faces = detect_faces(image)
    predictions = []

    for face in faces:
        embedding = get_face_embedding(image, face["box"])
        if embedding is None:
            continue
        name, confidence = find_match(embedding)
        predictions.append({
            "userid": name,
            "confidence": round(confidence * 100, 1),
            "x_min": face["box"]["x_min"],
            "y_min": face["box"]["y_min"],
            "x_max": face["box"]["x_max"],
            "y_max": face["box"]["y_max"],
        })

    elapsed = int((time.time() - start) * 1000)
    return jsonify({"success": True, "predictions": predictions, "processMs": elapsed})


@app.route("/v1/vision/face/list", methods=["POST", "GET"])
def list_faces():
    faces = list(_faces_db.keys())
    return jsonify({"success": True, "faces": faces, "processMs": 0})


@app.route("/v1/vision/face/delete", methods=["POST"])
def delete_face():
    userid = request.form.get("userid")
    if not userid:
        return jsonify({"success": False, "error": "userid required"}), 400

    if userid in _faces_db:
        # Remove embedding files
        for i in range(len(_faces_db[userid])):
            fpath = EMBEDDINGS_DIR / f"{userid}_{i}.npy"
            if fpath.exists():
                fpath.unlink()
        del _faces_db[userid]
        save_faces_db()
        return jsonify({"success": True, "message": f"face {userid} deleted"})
    return jsonify({"success": False, "error": f"face {userid} not found"})


@app.route("/v1/status", methods=["GET"])
def get_status():
    return jsonify({
        "success": True,
        "state": "running",
        "coral_tpu": _detect_interpreter is not None,
        "faces_count": len(_faces_db),
        "faces": list(_faces_db.keys()),
    })


if __name__ == "__main__":
    log.info("Starting Coral Face Recognition Server")
    init_detect_interpreter()
    init_embed_interpreter()
    load_faces_db()
    log.info(f"Registered faces: {list(_faces_db.keys())}")
    app.run(host="0.0.0.0", port=5100, debug=False)
