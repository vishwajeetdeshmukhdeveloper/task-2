"""
Image Text Extractor using Tesseract OCR and pytesseract
Efficiently extracts text from images using optical character recognition
"""

import pytesseract
from PIL import Image
from typing import Optional
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


class ImageExtractor:
    """Extract text from images using OCR (Tesseract)"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    def __init__(self, image_path: str):
        """
        Initialize image extractor
        
        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.extraction_results = {
            "file": image_path,
            "extraction_time": datetime.now().isoformat()
        }
        self._validate_image()
    
    def _validate_image(self) -> None:
        """Validate that file is a supported image format"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        file_ext = os.path.splitext(self.image_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {file_ext}")
    
    def extract_text(self, lang: str = 'eng', psm: int = 3) -> str:
        """
        Extract text from image using OCR
        
        Args:
            lang: Language for OCR (default: 'eng' for English)
                  For multiple languages: 'eng+fra+deu'
            psm: Page segmentation mode (default: 3 - automatic page segmentation)
                0 = Orientation and script detection
                1 = Automatic page segmentation with OSD
                3 = Fully automatic page segmentation
                6 = Assume single uniform block of text
                
        Returns:
            Extracted text from image
        """
        try:
            image = Image.open(self.image_path)
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=f'--psm {psm}'
            )
            
            self.extraction_results["text_extracted"] = True
            self.extraction_results["content_length"] = len(extracted_text)
            self.extraction_results["language"] = lang
            
            return extracted_text.strip()
        
        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            self.extraction_results["text_extracted"] = False
            self.extraction_results["error"] = str(e)
            return ""
    
    def extract_data(self, lang: str = 'eng') -> dict:
        """
        Extract data and metadata from image
        
        Args:
            lang: Language for OCR
            
        Returns:
            Dictionary with extracted data and metadata
        """
        try:
            image = Image.open(self.image_path)
            
            data = {
                "file": self.image_path,
                "image_size": image.size,
                "image_format": image.format,
                "image_mode": image.mode,
                "text": self.extract_text(lang=lang),
                "extraction_time": datetime.now().isoformat()
            }
            
            return data
        
        except Exception as e:
            print(f"Error extracting data from image: {str(e)}")
            return {"file": self.image_path, "error": str(e)}
    
    def extract_with_preprocessing(self, enhancement: str = 'contrast') -> str:
        """
        Extract text with image preprocessing for better OCR accuracy
        
        Args:
            enhancement: Type of enhancement ('contrast', 'grayscale', 'sharp', 'all')
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(self.image_path)
            
            if enhancement in ('grayscale', 'all'):
                image = image.convert('L')
            
            if enhancement in ('contrast', 'all'):
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2)
            
            if enhancement in ('sharp', 'all'):
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2)
            
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text.strip()
        
        except Exception as e:
            print(f"Error with preprocessing: {str(e)}")
            return self.extract_text()
    
    def save_text_to_file(self, output_path: str, lang: str = 'eng') -> bool:
        """
        Save extracted text to file
        
        Args:
            output_path: Path to save the output text file
            lang: Language for OCR
            
        Returns:
            True if successful, False otherwise
        """
        try:
            text = self.extract_text(lang=lang)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"Image OCR Extraction Report\n")
                f.write(f"Source: {self.image_path}\n")
                f.write(f"Extraction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write extracted text
                f.write("EXTRACTED TEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(text if text else "No text detected in image.\n")
            
            print(f"Successfully saved extracted text to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error saving text to file: {str(e)}")
            return False


def pick_image_file() -> Optional[str]:
    """
    Open a file picker dialog to select an image file
    Falls back to terminal-based selection if GUI is unavailable
    
    Returns:
        Path to selected image file, or None if cancelled
    """
    try:
        # Try GUI file picker first
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)  # Keep on top
        
        filetypes = (
            ('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif'),
            ('JPEG', '*.jpg *.jpeg'),
            ('PNG', '*.png'),
            ('BMP', '*.bmp'),
            ('TIFF', '*.tiff'),
            ('WebP', '*.webp'),
            ('GIF', '*.gif'),
            ('All Files', '*.*')
        )
        
        file_path = filedialog.askopenfilename(
            title='Select an Image File',
            filetypes=filetypes,
            initialdir=os.path.expanduser('~')
        )
        
        root.destroy()
        return file_path if file_path else None
    
    except Exception as e:
        print(f"\n[GUI Picker Failed] {str(e)}")
        print("Falling back to terminal-based file selection...\n")
        return pick_image_file_terminal()


def pick_image_file_terminal() -> Optional[str]:
    """
    Terminal-based file picker for image files
    Lists images in current directory and allows selection
    
    Returns:
        Path to selected image file, or None if cancelled
    """
    # First, try current directory
    current_dir = os.getcwd()
    
    print(f"Current directory: {current_dir}\n")
    print("Searching for image files...\n")
    
    # Find all image files in current directory and subdirectories
    image_files = []
    for root_dir, dirs, files in os.walk(current_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ImageExtractor.SUPPORTED_FORMATS:
                full_path = os.path.join(root_dir, file)
                image_files.append(full_path)
    
    if not image_files:
        print("No image files found in current directory or subdirectories.")
        manual_path = input("Enter image file path manually (or press Enter to cancel): ").strip()
        
        if manual_path:
            if os.path.exists(manual_path) and os.path.splitext(manual_path)[1].lower() in ImageExtractor.SUPPORTED_FORMATS:
                return manual_path
            else:
                print(f"Error: File not found or unsupported format: {manual_path}")
                return None
        return None
    
    # Display found images
    print("Found image files:\n")
    for idx, file in enumerate(image_files, 1):
        print(f"  [{idx}] {file}")
    
    print(f"\n  [0] Enter path manually")
    print(f"  [C] Cancel\n")
    
    # Get user selection
    while True:
        choice = input("Select an image (enter number): ").strip().lower()
        
        if choice == 'c':
            print("Selection cancelled.")
            return None
        
        if choice == '0':
            manual_path = input("Enter image file path: ").strip()
            if manual_path and os.path.exists(manual_path):
                if os.path.splitext(manual_path)[1].lower() in ImageExtractor.SUPPORTED_FORMATS:
                    return manual_path
                else:
                    print("Error: Unsupported format")
            else:
                print("Error: File not found")
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                selected = image_files[idx]
                print(f"\nSelected: {selected}\n")
                return selected
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(image_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def extract_image(image_path: str, output_path: Optional[str] = None, lang: str = 'eng') -> str:
    """
    Convenience function to extract text from image and save to file
    
    Args:
        image_path: Path to image file
        output_path: Optional output file path (auto-generated if not provided)
        lang: Language for OCR
        
    Returns:
        Extracted text content
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    extractor = ImageExtractor(image_path)
    text = extractor.extract_text(lang=lang)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{base_name}_extracted.txt"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    extractor.save_text_to_file(output_path, lang=lang)
    
    return text


def extract_text_quick():
    """
    Quick extraction with file picker
    Automatically uses best enhancement settings for maximum accuracy
    """
    print("\n" + "=" * 80)
    print("QUICK IMAGE TEXT EXTRACTION (BEST QUALITY)")
    print("=" * 80)
    
    # Step 1: Pick an image file
    print("\nSelecting image file...\n")
    image_path = pick_image_file()
    
    if not image_path:
        print("\nNo file selected. Exiting.")
        return
    
    print(f"✓ Selected: {image_path}")
    
    # Extract with high settings (all enhancements)
    print("\n" + "=" * 80)
    print("EXTRACTING TEXT (BEST SETTINGS)...")
    print("=" * 80 + "\n")
    
    try:
        extractor = ImageExtractor(image_path)
        
        # Use 'all' enhancement for best accuracy
        print("Applying all enhancements (grayscale + contrast + sharpness)...")
        text = extractor.extract_with_preprocessing(enhancement='all')
        
        # Save to file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output/{base_name}_extracted_{timestamp}.txt"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Image OCR Extraction Report\n")
            f.write(f"Source: {image_path}\n")
            f.write(f"Enhancement: All (Grayscale + Contrast + Sharpness)\n")
            f.write(f"Extraction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write("EXTRACTED TEXT:\n")
            f.write("-" * 80 + "\n")
            f.write(text if text else "No text detected in image.\n")
        
        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n✓ Output saved to: {output_path}")
        print(f"✓ Extracted {len(text)} characters")
        print(f"✓ Enhancement: All (Best Quality)")
        
        print("\nExtracted Text Preview:")
        print("-" * 80)
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(preview if preview.strip() else "[No text detected]")
        print("-" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")


def main():
    """
    Interactive main function for image text extraction
    Prompts user to select an image file and extracts text
    """
    print("\n" + "=" * 80)
    print("IMAGE TEXT EXTRACTION - FILE PICKER")
    print("=" * 80)
    
    # Step 1: Pick an image file
    print("\nStep 1: Select an image file\n")
    image_path = pick_image_file()
    
    if not image_path:
        print("\nNo file selected. Exiting.")
        return
    
    print(f"\n✓ Selected: {image_path}")
    
    # Step 2: Ask for enhancement preference
    print("\nStep 2: Choose preprocessing enhancement:")
    print("  [1] No preprocessing (faster)")
    print("  [2] Contrast enhancement")
    print("  [3] Grayscale conversion")
    print("  [4] Sharpness enhancement")
    print("  [5] All enhancements (best accuracy)")
    
    enhancement_choice = input("\nEnter your choice (1-5) [default: 1]: ").strip()
    
    enhancement_map = {
        '1': None,
        '2': 'contrast',
        '3': 'grayscale',
        '4': 'sharp',
        '5': 'all'
    }
    
    enhancement = enhancement_map.get(enhancement_choice, None)
    
    # Step 3: Ask for language
    print("\nStep 3: Choose OCR language:")
    print("  [1] English (default)")
    print("  [2] English + French")
    print("  [3] English + Spanish + German")
    print("  [4] Custom (enter language codes)")
    
    lang_choice = input("\nEnter your choice (1-4) [default: 1]: ").strip()
    
    lang_map = {
        '1': 'eng',
        '2': 'eng+fra',
        '3': 'eng+spa+deu',
    }
    
    if lang_choice == '4':
        lang = input("Enter language codes (e.g., eng+fra+deu): ").strip() or 'eng'
    else:
        lang = lang_map.get(lang_choice, 'eng')
    
    print(f"\n✓ Language set to: {lang}")
    
    # Step 4: Extract with chosen settings
    print("\n" + "=" * 80)
    print("EXTRACTING TEXT...")
    print("=" * 80)
    
    try:
        extractor = ImageExtractor(image_path)
        
        # Extract with appropriate method
        if enhancement:
            print(f"Applying {enhancement} enhancement...")
            text = extractor.extract_with_preprocessing(enhancement=enhancement)
        else:
            print("Extracting without preprocessing...")
            text = extractor.extract_text(lang=lang)
        
        # Step 5: Save to file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output/{base_name}_extracted_{timestamp}.txt"
        
        extractor.save_text_to_file(output_path, lang=lang)
        
        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n✓ Output saved to: {output_path}")
        print(f"✓ Extracted {len(text)} characters")
        print(f"✓ Language: {lang}")
        print(f"✓ Enhancement: {enhancement or 'None'}")
        
        print("\nExtracted Text Preview:")
        print("-" * 80)
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(preview)
        print("-" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    extract_text_quick()
