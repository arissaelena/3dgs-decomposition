#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
URLS_FILE="data_urls.txt"

if [ ! -f "$URLS_FILE" ]; then
    exit 1
fi

echo "==> Downloading datasets..."
wget -nc -q --show-progress -i "$URLS_FILE"

echo "==> Extracting Hypersim scenes..."
mkdir -p hypersim
shopt -s nullglob
for zip_file in ai_*.zip; do
    echo "   Unzipping $zip_file..."
    python3 -m zipfile -e "$zip_file" hypersim/
    rm -f "$zip_file"
done
shopt -u nullglob

echo "==> Extracting Mip-NeRF 360 scenes..."
mkdir -p mipnerf360
if [ -f "360_v2.zip" ]; then
    python3 -c '
import zipfile
scenes = ("bicycle/", "garden/", "room/")
with zipfile.ZipFile("360_v2.zip") as file:
    members = [name for name in file.namelist() if name.startswith(scenes)]
    file.extractall("mipnerf360", members)
'
    rm -f "360_v2.zip"
fi

echo "DONE"
