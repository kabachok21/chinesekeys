FROM python:3.12-slim

# Tesseract OCR engine (the chi_sim language pack ships in this repo's own
# tessdata/ folder and is picked up automatically - see
# ocr/recognize.py:_tessdata_dir_config - so only the engine binary is
# needed here, not the tesseract-ocr-chi-sim apt package).
# libgl1 + libglib2.0-0: opencv-python-headless still dynamically links
# against libGL/libglib at import time on Debian, even though it has no
# GUI of its own.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# main.py defaults to the waitress production server (no FLASK_DEBUG set)
# and reads the listen port from $PORT, which host platforms set for you.
CMD ["python", "main.py"]
