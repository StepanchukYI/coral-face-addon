# Coral Face Recognition

Home Assistant addon for face detection and recognition using Google Coral USB TPU.

DeepStack-compatible REST API — works with [Double-Take](https://github.com/jakowenko/double-take) out of the box.

## Features

- Face detection on Coral USB TPU (MobileNet SSD v2 Face)
- Face embedding via MobileFaceNet (TFLite, CPU)
- DeepStack-compatible API (`/v1/vision/face/*`)
- Face database with register/recognize/list/delete
- Cosine similarity matching with configurable thresholds

## Requirements

- Home Assistant OS (HAOS) on Raspberry Pi 4 (aarch64)
- Google Coral USB Accelerator

## Installation

Add this repository to Home Assistant:

1. Go to **Settings** > **Add-ons** > **Add-on Store**
2. Click **...** (top right) > **Repositories**
3. Add: `https://github.com/StepanchukYI/coral-face-addon`
4. Install **Coral Face Recognition**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/vision/face/register` | Register a face (form: `userid`, `image`) |
| POST | `/v1/vision/face/recognize` | Recognize faces in image (form: `image`) |
| GET/POST | `/v1/vision/face/list` | List registered faces |
| POST | `/v1/vision/face/delete` | Delete a face (form: `userid`) |
| GET | `/v1/status` | Server status, TPU info, face count |

## Usage with Double-Take

```yaml
# double-take config.yml
detectors:
  deepstack:
    url: http://<HA_IP>:5100
    key: ""
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECT_THRESHOLD` | `0.5` | Minimum confidence for face detection |
| `RECOGNIZE_THRESHOLD` | `0.45` | Minimum cosine similarity for recognition |
| `DATA_DIR` | `/data` | Face database storage path |

## Architecture

```
Image → Coral TPU (face detection) → MobileFaceNet (embedding) → cosine similarity → match
```

- Detection: `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite` (Coral TPU)
- Embedding: `mobilefacenet.tflite` (CPU, TFLite runtime)
- Database: JSON index + numpy `.npy` files in `/data/`
