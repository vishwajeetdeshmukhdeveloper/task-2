"""
PDF Text and Data Extractor using pdfplumber
Efficiently extracts text, tables, and data from PDF files
"""

import pdfplumber
from typing import List, Dict, Optional
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


class PDFExtractor:
    """Extract text and structured data from PDF files"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.extraction_results = {
            "file": pdf_path,
            "extraction_time": datetime.now().isoformat(),
            "pages": []
        }
    
    def extract_all_text(self) -> str:
        """
        Extract all text from PDF
        
        Returns:
            Combined text from all pages
        """
        all_text = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        all_text.append(f"--- Page {page_num} ---\n{text}")
                        self.extraction_results["pages"].append({
                            "page_number": page_num,
                            "text_extracted": True,
                            "content_length": len(text)
                        })
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
        
        return "\n\n".join(all_text)
    
    def extract_tables(self) -> List[Dict]:
        """
        Extract tables from PDF
        
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table in enumerate(page_tables, 1):
                            tables.append({
                                "page": page_num,
                                "table_index": table_idx,
                                "data": table
                            })
        except Exception as e:
            print(f"Error extracting tables from PDF: {str(e)}")
        
        return tables
    
    def extract_structured_data(self) -> Dict:
        """
        Extract all structured data from PDF (text + tables + metadata)
        
        Returns:
            Dictionary containing all extracted data
        """
        structured_data = {
            "file": self.pdf_path,
            "metadata": self._extract_metadata(),
            "text": self.extract_all_text(),
            "tables": self.extract_tables()
        }
        
        return structured_data
    
    def _extract_metadata(self) -> Dict:
        """Extract PDF metadata"""
        metadata = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)
                metadata["document_info"] = pdf.metadata or {}
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def save_text_to_file(self, output_path: str, include_tables: bool = True) -> bool:
        """
        Save extracted text to file
        
        Args:
            output_path: Path to save the output text file
            include_tables: Whether to include table data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"PDF Extraction Report\n")
                f.write(f"Source: {self.pdf_path}\n")
                f.write(f"Extraction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write extracted text
                f.write("EXTRACTED TEXT:\n")
                f.write("-" * 80 + "\n")
                text = self.extract_all_text()
                f.write(text if text else "No text content found.\n")
                f.write("\n\n")
                
                # Write tables if requested
                if include_tables:
                    f.write("EXTRACTED TABLES:\n")
                    f.write("-" * 80 + "\n")
                    tables = self.extract_tables()
                    if tables:
                        for table_info in tables:
                            f.write(f"\nPage {table_info['page']}, Table {table_info['table_index']}:\n")
                            for row in table_info['data']:
                                f.write(" | ".join(str(cell) if cell else "" for cell in row) + "\n")
                    else:
                        f.write("No tables found.\n")
                
                print(f"Successfully saved extracted text to {output_path}")
                return True
        except Exception as e:
            print(f"Error saving text to file: {str(e)}")
            return False


def extract_pdf(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Convenience function to extract PDF and save to file
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output file path (auto-generated if not provided)
        
    Returns:
        Extracted text content
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    extractor = PDFExtractor(pdf_path)
    text = extractor.extract_all_text()
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"output/{base_name}_extracted.txt"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    extractor.save_text_to_file(output_path, include_tables=True)
    
    return text


def pick_pdf_file() -> Optional[str]:
    """
    Open a file picker dialog to select a PDF file
    Falls back to terminal-based selection if GUI is unavailable
    
    Returns:
        Path to selected PDF file, or None if cancelled
    """
    try:
        # Try GUI file picker first
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)  # Keep on top
        
        filetypes = (
            ('PDF Files', '*.pdf'),
            ('All Files', '*.*')
        )
        
        file_path = filedialog.askopenfilename(
            title='Select a PDF File',
            filetypes=filetypes,
            initialdir=os.path.expanduser('~')
        )
        
        root.destroy()
        return file_path if file_path else None
    
    except Exception as e:
        print(f"\n[GUI Picker Failed] {str(e)}")
        print("Falling back to terminal-based file selection...\n")
        return pick_pdf_file_terminal()


def pick_pdf_file_terminal() -> Optional[str]:
    """
    Terminal-based file picker for PDF files
    Lists PDFs in current directory and allows selection
    
    Returns:
        Path to selected PDF file, or None if cancelled
    """
    # First, try current directory
    current_dir = os.getcwd()
    
    print(f"Current directory: {current_dir}\n")
    print("Searching for PDF files...\n")
    
    # Find all PDF files in current directory and subdirectories
    pdf_files = []
    for root_dir, dirs, files in os.walk(current_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root_dir, file)
                pdf_files.append(full_path)
    
    if not pdf_files:
        print("No PDF files found in current directory or subdirectories.")
        manual_path = input("Enter PDF file path manually (or press Enter to cancel): ").strip()
        
        if manual_path:
            if os.path.exists(manual_path) and manual_path.lower().endswith('.pdf'):
                return manual_path
            else:
                print(f"Error: File not found or not a PDF: {manual_path}")
                return None
        return None
    
    # Display found PDFs
    print("Found PDF files:\n")
    for idx, file in enumerate(pdf_files, 1):
        print(f"  [{idx}] {file}")
    
    print(f"\n  [0] Enter path manually")
    print(f"  [C] Cancel\n")
    
    # Get user selection
    while True:
        choice = input("Select a PDF (enter number): ").strip().lower()
        
        if choice == 'c':
            print("Selection cancelled.")
            return None
        
        if choice == '0':
            manual_path = input("Enter PDF file path: ").strip()
            if manual_path and os.path.exists(manual_path):
                if manual_path.lower().endswith('.pdf'):
                    return manual_path
                else:
                    print("Error: Not a PDF file")
            else:
                print("Error: File not found")
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(pdf_files):
                selected = pdf_files[idx]
                print(f"\nSelected: {selected}\n")
                return selected
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(pdf_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def extract_pdf_quick():
    """
    Quick PDF extraction with file picker
    Automatically extracts text and tables with best settings
    """
    print("\n" + "=" * 80)
    print("QUICK PDF TEXT EXTRACTION (WITH TABLES)")
    print("=" * 80)
    
    # Step 1: Pick a PDF file
    print("\nSelecting PDF file...\n")
    pdf_path = pick_pdf_file()
    
    if not pdf_path:
        print("\nNo file selected. Exiting.")
        return
    
    print(f"✓ Selected: {pdf_path}")
    
    # Extract with all features
    print("\n" + "=" * 80)
    print("EXTRACTING PDF...")
    print("=" * 80 + "\n")
    
    try:
        extractor = PDFExtractor(pdf_path)
        
        # Get metadata
        metadata = extractor._extract_metadata()
        print(f"Total pages: {metadata.get('total_pages', 'Unknown')}")
        
        # Extract text
        print("Extracting text...")
        text = extractor.extract_all_text()
        
        # Extract tables
        print("Extracting tables...")
        tables = extractor.extract_tables()
        
        # Save to file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output/{base_name}_extracted_{timestamp}.txt"
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"PDF Extraction Report\n")
            f.write(f"Source: {pdf_path}\n")
            f.write(f"Total Pages: {metadata.get('total_pages', 'Unknown')}\n")
            f.write(f"Extraction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write extracted text
            f.write("EXTRACTED TEXT:\n")
            f.write("-" * 80 + "\n")
            f.write(text if text else "No text content found.\n")
            f.write("\n\n")
            
            # Write tables
            f.write("EXTRACTED TABLES:\n")
            f.write("-" * 80 + "\n")
            if tables:
                for table_info in tables:
                    f.write(f"\nPage {table_info['page']}, Table {table_info['table_index']}:\n")
                    for row in table_info['data']:
                        f.write(" | ".join(str(cell) if cell else "" for cell in row) + "\n")
            else:
                f.write("No tables found.\n")
        
        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\n✓ Output saved to: {output_path}")
        print(f"✓ Extracted {len(text)} characters from text")
        print(f"✓ Found {len(tables)} table(s)")
        print(f"✓ Pages processed: {metadata.get('total_pages', 'Unknown')}")
        
        print("\nExtracted Text Preview:")
        print("-" * 80)
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(preview if preview.strip() else "[No text detected]")
        print("-" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")


if __name__ == "__main__":
    extract_pdf_quick()
