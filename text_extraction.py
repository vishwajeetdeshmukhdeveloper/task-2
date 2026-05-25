"""
Unified Text Extraction Module
Extracts text from both PDFs and Images in a single interface
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import pdfplumber
from PIL import Image
import pytesseract


class TextExtractor:
    """Unified text extraction for PDFs and Images"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_pdf(self, file_path: str) -> Dict:
        """Extract text from PDF file"""
        try:
            text = ""
            tables = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            tables.append({
                                'page': page_num,
                                'data': table
                            })
            
            return {
                'success': True,
                'file_type': 'pdf',
                'file_path': file_path,
                'text': text.strip(),
                'tables': tables,
                'page_count': len(pdf.pages),
                'extracted_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'file_type': 'pdf',
                'file_path': file_path,
                'error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
    
    def extract_image(self, file_path: str, lang: str = 'eng', psm: int = 3) -> Dict:
        """Extract text from image file using OCR"""
        try:
            # Preprocess image for better OCR
            image = Image.open(file_path)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Extract text using Tesseract OCR
            config = f'--psm {psm}'
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            
            return {
                'success': True,
                'file_type': 'image',
                'file_path': file_path,
                'text': text.strip(),
                'image_size': image.size,
                'extracted_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'file_type': 'image',
                'file_path': file_path,
                'error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
    
    def extract_file(self, file_path: str) -> Dict:
        """
        Auto-detect file type and extract text
        
        Args:
            file_path: Path to PDF or image file
        
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'File not found: {file_path}',
                'extracted_at': datetime.now().isoformat()
            }
        
        file_ext = Path(file_path).suffix.lower()
        
        # PDF extraction
        if file_ext == '.pdf':
            return self.extract_pdf(file_path)
        
        # Image extraction
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return self.extract_image(file_path)
        
        else:
            return {
                'success': False,
                'file_path': file_path,
                'error': f'Unsupported file type: {file_ext}',
                'extracted_at': datetime.now().isoformat()
            }
    
    def extract_from_directory(self, directory: str, recursive: bool = False) -> list:
        """
        Extract text from all supported files in directory
        
        Args:
            directory: Path to directory containing files
            recursive: Whether to search subdirectories
        
        Returns:
            List of extraction results
        """
        results = []
        
        if not os.path.exists(directory):
            return [{'success': False, 'error': f'Directory not found: {directory}'}]
        
        # Supported extensions
        extensions = ('*.pdf', '*.jpg', '*.jpeg', '*.png', '*.bmp')
        
        # Find files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(Path(directory).rglob(ext))
        else:
            files = []
            for ext in extensions:
                files.extend(Path(directory).glob(ext))
        
        # Extract from each file
        for file_path in sorted(files):
            result = self.extract_file(str(file_path))
            results.append(result)
        
        return results
    
    def save_extraction(self, extraction_result: Dict, filename: Optional[str] = None) -> str:
        """
        Save extracted text to file
        
        Args:
            extraction_result: Result from extract_file()
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if not extraction_result['success']:
            return None
        
        # Generate filename if not provided
        if filename is None:
            original_name = Path(extraction_result['file_path']).stem
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{original_name}_extracted_{timestamp}.txt"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Write extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Extracted from: {extraction_result['file_path']}\n")
            f.write(f"File Type: {extraction_result['file_type']}\n")
            f.write(f"Extracted at: {extraction_result['extracted_at']}\n")
            f.write("=" * 100 + "\n\n")
            
            if extraction_result['file_type'] == 'pdf':
                f.write(f"Pages: {extraction_result['page_count']}\n\n")
            
            f.write(extraction_result['text'])
            
            if extraction_result.get('tables'):
                f.write("\n\n" + "=" * 100 + "\n")
                f.write("TABLES EXTRACTED\n")
                f.write("=" * 100 + "\n\n")
                for idx, table_info in enumerate(extraction_result['tables'], 1):
                    f.write(f"Table {idx} (Page {table_info['page']}):\n")
                    for row in table_info['data']:
                        f.write(str(row) + "\n")
                    f.write("\n")
        
        return output_path


# Convenience functions for quick usage
def extract(file_path: str, save: bool = True) -> Dict:
    """Quick extraction function"""
    extractor = TextExtractor()
    result = extractor.extract_file(file_path)
    
    if save and result['success']:
        extractor.save_extraction(result)
    
    return result


def extract_batch(directory: str, recursive: bool = False, save: bool = True) -> list:
    """Quick batch extraction function"""
    extractor = TextExtractor()
    results = extractor.extract_from_directory(directory, recursive=recursive)
    
    if save:
        for result in results:
            if result['success']:
                extractor.save_extraction(result)
    
    return results


if __name__ == "__main__":
    # Example usage
    import json
    
    # Extract from sample files
    samples_dir = "samples/pdf"
    
    if os.path.exists(samples_dir):
        print("Extracting from samples/...")
        results = extract_batch(samples_dir)
        
        for result in results:
            print(f"\n{'='*50}")
            print(f"File: {result['file_path']}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Text length: {len(result['text'])} characters")
                if result.get('page_count'):
                    print(f"Pages: {result['page_count']}")
            else:
                print(f"Error: {result.get('error')}")
    else:
        print(f"Samples directory not found: {samples_dir}")
