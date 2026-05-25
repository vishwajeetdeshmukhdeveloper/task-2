"""
Configuration settings for extraction system
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
SAMPLES_DIR = PROJECT_ROOT / 'samples'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Supported file types
SUPPORTED_PDF_FORMATS = {'.pdf'}
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

# OCR Configuration
OCR_LANGUAGE = 'eng'  # Default language (e.g., 'eng+fra' for English + French)
OCR_PAGE_SEGMENTATION = 3  # 0-13, default 3 (automatic)

# PDF Extraction Configuration
EXTRACT_TABLES = True
EXTRACT_TEXT = True

# Output Configuration
OUTPUT_ENCODING = 'utf-8'
INCLUDE_METADATA = True
INCLUDE_TIMESTAMPS = True

# Logging Configuration
LOG_FILE = OUTPUT_DIR / 'extraction_log.json'
ENABLE_LOGGING = True

# Feature Flags
ENABLE_IMAGE_PREPROCESSING = True
PREPROCESS_ENHANCEMENT = 'contrast'  # 'contrast', 'grayscale', 'sharp', 'all'

# Batch Processing
MAX_BATCH_SIZE = 100
