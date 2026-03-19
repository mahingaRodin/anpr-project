# ANPR Project - Car Number Plate Extraction System

This project implements a simple Automatic Number Plate Recognition (ANPR) pipeline based on the required assignment flow:

**Detection → Alignment → OCR → Validation → Temporal → Save**

## Features
- Captures frames from a webcam
- Detects a likely number plate region
- Aligns the plate using perspective correction
- Reads plate text using Tesseract OCR
- Validates OCR output using plate patterns
- Confirms plate text after multiple observations
- Saves confirmed plates into `data/plates.csv`

## Project Structure
```text
anpr-project/
├── README.md
├── requirements.txt
├── src/
│   ├── camera.py
│   ├── detect.py
│   ├── align.py
│   ├── ocr.py
│   ├── validate.py
│   ├── temporal.py
│   ├── storage.py
│   └── main.py
├── data/
│   └── plates.csv
└── screenshots/