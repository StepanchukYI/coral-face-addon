#!/usr/bin/env bash
set -e

# Read addon options from /data/options.json and export as env vars
if [ -f /data/options.json ]; then
    export DETECT_THRESHOLD=$(python3 -c "import json; print(json.load(open('/data/options.json')).get('detect_threshold', 0.5))")
    export RECOGNIZE_THRESHOLD=$(python3 -c "import json; print(json.load(open('/data/options.json')).get('recognize_threshold', 0.45))")
    export PORT=$(python3 -c "import json; print(json.load(open('/data/options.json')).get('port', 5100))")
fi

echo "Starting Coral Face Recognition Server..."
echo "  DETECT_THRESHOLD=${DETECT_THRESHOLD:-0.5}"
echo "  RECOGNIZE_THRESHOLD=${RECOGNIZE_THRESHOLD:-0.45}"
echo "  PORT=${PORT:-5100}"

cd /opt/coral-face
exec python3 server.py
