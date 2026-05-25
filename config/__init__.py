"""
Configuration module
"""

from .settings import *
from .utils import *

__all__ = [
    'PROJECT_ROOT',
    'OUTPUT_DIR',
    'SAMPLES_DIR',
    'CONFIG_DIR',
    'SUPPORTED_PDF_FORMATS',
    'SUPPORTED_IMAGE_FORMATS',
    'OCR_LANGUAGE',
    'OCR_PAGE_SEGMENTATION',
    'get_supported_files',
    'validate_file',
    'create_output_filename',
]
