ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.11-bookworm
FROM ${BUILD_FROM}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    libusb-1.0-0 \
    libc6 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Add Coral Edge TPU repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/coral-face

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download face detection model (Coral Edge TPU)
RUN mkdir -p models && \
    curl -fsSL -o models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
    "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

# Download MobileFaceNet embedding model (TFLite CPU)
# Using a pre-trained MobileFaceNet converted to TFLite
RUN curl -fsSL -o models/mobilefacenet.tflite \
    "https://github.com/nicehuster/mobilefacenet-tflite/raw/main/model/mobilefacenet.tflite" \
    || curl -fsSL -o models/mobilefacenet.tflite \
    "https://raw.githubusercontent.com/nicehuster/mobilefacenet-tflite/main/model/mobilefacenet.tflite"

COPY server.py .

EXPOSE 5100

CMD ["python3", "server.py"]
