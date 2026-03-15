#!/usr/bin/env bash
set -e

echo "Starting Coral Face Recognition Server..."
cd /opt/coral-face
exec python3 server.py
