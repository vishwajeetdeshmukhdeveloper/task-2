# PDF & Image Text Extraction System

A robust Python system for extracting text and data from PDF files and images using industry-leading libraries.

## Features

- **PDF Extraction**: Uses `pdfplumber` for efficient text and table extraction
- **Image OCR**: Uses `pytesseract` with Tesseract engine for optical character recognition
- **Batch Processing**: Extract from entire directories with recursive support
- **Table Detection**: Automatically detects and extracts tables from PDFs
- **Image Preprocessing**: Optional image enhancement for better OCR accuracy
- **Comprehensive Logging**: Track all extraction operations with detailed logs
- **Text Output**: Saves all extracted text to organized text files

## Supported File Formats

### PDFs
- `.pdf` - All standard PDF formats

### Images
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images
- `.bmp` - Bitmap images
- `.tiff` - TIFF images
- `.webp` - WebP images
- `.gif` - GIF images

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine (for image text extraction)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (Required for image extraction)

#### Windows
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run: `tesseract-ocr-w64-setup-v5.x.x.exe`
3. During installation, note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Add to your Python code or environment:
   ```python
   import pytesseract
   pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr
```

#### Linux (Fedora/CentOS)
```bash
sudo yum install tesseract
```

## Usage

### Quick Start

#### 1. Extract from a Single PDF
```python
from extraction_system import ExtractionSystem

system = ExtractionSystem(output_dir='output')
success, message = system.extract_file('path/to/document.pdf')
print(message)
```

#### 2. Extract from a Single Image
```python
from extraction_system import ExtractionSystem

system = ExtractionSystem(output_dir='output')
success, message = system.extract_file('path/to/image.png')
print(message)
```

#### 3. Batch Extract from Directory
```python
from extraction_system import ExtractionSystem

system = ExtractionSystem(output_dir='output')
results = system.extract_directory('samples/', recursive=True)
system.print_summary()
system.save_extraction_log('output/extraction_log.json')
```

#### 4. Extract Multiple Specific Files
```python
from extraction_system import ExtractionSystem

system = ExtractionSystem(output_dir='output')

files = ['document1.pdf', 'image1.png', 'document2.pdf']
for file_path in files:
    success, message = system.extract_file(file_path)
    print(f"{'✓' if success else '✗'} {message}")
```

### Using Individual Extractors

#### PDF Extraction
```python
from extractors.pdf_extractor import PDFExtractor

# Extract text
extractor = PDFExtractor('document.pdf')
text = extractor.extract_all_text()
print(text)

# Save to file
extractor.save_text_to_file('output/document.txt', include_tables=True)

# Extract tables
tables = extractor.extract_tables()
for table in tables:
    print(f"Table on page {table['page']}")
```

#### Image Extraction with OCR
```python
from extractors.image_extractor import ImageExtractor

# Basic extraction
extractor = ImageExtractor('image.png')
text = extractor.extract_text()
print(text)

# Extract with preprocessing (better for low-quality images)
text = extractor.extract_with_preprocessing(enhancement='all')

# Save to file
extractor.save_text_to_file('output/image.txt', lang='eng')

# Multiple languages
text = extractor.extract_text(lang='eng+fra')  # English + French
```

### Run the Example Script
```bash
python main.py
```

## Project Structure

```
.
├── config/
│   ├── settings.py          # Configuration settings
│   └── utils.py             # Utility functions
├── extractors/
│   ├── __init__.py          # Package initialization
│   ├── pdf_extractor.py     # PDF extraction module
│   └── image_extractor.py   # Image OCR extraction module
├── output/                   # Extracted text files (auto-created)
├── samples/                  # Sample files for testing
├── extraction_system.py      # Main orchestrator
├── main.py                   # Example usage script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Configuration

Edit `config/settings.py` to customize:

- **OCR_LANGUAGE**: Default language for OCR (e.g., 'eng', 'fra', 'eng+fra')
- **OCR_PAGE_SEGMENTATION**: Page segmentation mode (0-13)
- **EXTRACT_TABLES**: Whether to extract tables from PDFs
- **OUTPUT_ENCODING**: Output file encoding (default: utf-8)
- **ENABLE_IMAGE_PREPROCESSING**: Enable image enhancement for OCR

## Available OCR Languages

Tesseract supports 100+ languages. Common ones:
- `eng` - English
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `por` - Portuguese
- `rus` - Russian
- `jpn` - Japanese
- `chi_sim` - Chinese (Simplified)

Use multiple languages: `'eng+fra+deu'`

## API Reference

### ExtractionSystem

Main orchestrator class for batch and single file extraction.

```python
system = ExtractionSystem(output_dir='output')

# Extract single file
success, message = system.extract_file(file_path, output_file=None)

# Extract directory
results = system.extract_directory(directory_path, recursive=True)

# Get logs
log = system.get_extraction_log()
system.save_extraction_log('log.json')

# Print summary
system.print_summary()
```

### PDFExtractor

Direct PDF extraction.

```python
extractor = PDFExtractor('file.pdf')

# Get text
text = extractor.extract_all_text()

# Get tables
tables = extractor.extract_tables()

# Save to file
extractor.save_text_to_file('output.txt', include_tables=True)

# Get all data
data = extractor.extract_structured_data()
```

### ImageExtractor

Direct image OCR extraction.

```python
extractor = ImageExtractor('image.png')

# Get text
text = extractor.extract_text(lang='eng', psm=3)

# With preprocessing
text = extractor.extract_with_preprocessing(enhancement='contrast')

# Get image data
data = extractor.extract_data(lang='eng')

# Save to file
extractor.save_text_to_file('output.txt', lang='eng')
```

## Output Format

Extracted text files contain:
1. **Header** - Source file, extraction timestamp
2. **Extracted Text** - All text content from the file
3. **Tables** (PDFs only) - Formatted table data

Example:
```
PDF Extraction Report
Source: document.pdf
Extraction Time: 2026-05-15 14:30:45
================================================================================

EXTRACTED TEXT:
--------------------------------------------------------------------------------
[Full text content...]

EXTRACTED TABLES:
--------------------------------------------------------------------------------
Page 1, Table 1:
Header 1 | Header 2 | Header 3
Value 1  | Value 2  | Value 3
```

## Troubleshooting

### Tesseract Not Found (Image Extraction)
```python
import pytesseract
# Set path before importing extractors
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Poor OCR Results
- Use `extract_with_preprocessing()` with enhancement
- Try different PSM modes: 0-13
- Ensure image quality is reasonable
- Try specifying correct language

### PDF Extraction Issues
- Ensure PDF is not corrupted
- Some PDFs may have text as images (use image extraction instead)
- Check PDF permissions

### Memory Issues with Large Files
- Process files in batches
- Use `extract_directory()` with smaller directories

## Performance Tips

1. **Batch Processing**: Use `extract_directory()` for multiple files
2. **Language Specification**: Specify only required languages for OCR
3. **Image Quality**: Preprocess images before extraction
4. **Multithreading**: Implement threading for batch operations (custom)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pdfplumber | 0.10.3 | PDF text and table extraction |
| pytesseract | 0.3.10 | OCR interface |
| Pillow | 10.1.0 | Image processing |
| PyPDF2 | 4.1.1 | Alternative PDF handling |
| python-dotenv | 1.0.0 | Configuration management |

## License

This project is provided as-is for text extraction purposes.

## Contributing

Feel free to extend this system with:
- Support for additional file formats
- Advanced image preprocessing
- Parallel processing
- Web interface
- Database integration

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example script (`main.py`)
3. Check configuration in `config/settings.py`
4. Review extraction logs in `output/`

## Version History

- **v1.0** - Initial release with PDF and image extraction
  - PDF text extraction with table detection
  - Image OCR with Tesseract
  - Batch processing support
  - Comprehensive logging
