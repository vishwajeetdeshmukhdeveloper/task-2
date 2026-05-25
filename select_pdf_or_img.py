#!/usr/bin/env python
"""
Interactive PDF/Image Upload with File Picker
Allows users to select files from their system using a GUI file picker
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import requests
import json
from datetime import datetime

# Import API client
from api_client import MedicalAPIClient


def show_welcome():
    """Display welcome message"""
    print("\n" + "="*80)
    print("📄 MEDICAL DOCUMENT UPLOAD & PREDICTION")
    print("="*80)
    print("\n✓ File Picker will open in a moment...")
    print("✓ Select a PDF or image file to process")
    print("✓ Supported formats: PDF, JPG, PNG, BMP (Max 50MB)")
    print("\n" + "="*80 + "\n")


def pick_file():
    """Open file picker dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    file_path = filedialog.askopenfilename(
        title="Select Medical Document",
        filetypes=[
            ("All Documents", "*.pdf *.jpg *.jpeg *.png *.bmp"),
            ("PDF Files", "*.pdf"),
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def validate_file(file_path):
    """Validate selected file"""
    if not file_path:
        return False, "No file selected"
    
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB
        return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max 50MB)"
    
    supported_ext = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in supported_ext:
        return False, f"Unsupported format: {file_ext}"
    
    return True, "Valid"


def process_file(file_path, option):
    """Process the selected file"""
    print(f"\n📂 Selected File: {file_path}")
    print(f"📊 File Size: {os.path.getsize(file_path) / 1024:.1f} KB")
    print("\n" + "-"*80)
    print("⏳ Processing... Please wait\n")
    
    client = MedicalAPIClient()
    
    try:
        if option == "1":
            # Extract only
            print("🔍 Extracting text...")
            result = client.extract(file_path)
            
            if result['success']:
                print("✅ Extraction successful!\n")
                print("📝 EXTRACTED TEXT:")
                print("-" * 80)
                text = result['text']
                if len(text) > 500:
                    print(text[:500] + "\n... [truncated]")
                else:
                    print(text)
                print("-" * 80)
                
        elif option == "2":
            # Predict only
            print("🔍 Extracting text & generating prediction...")
            result = client.pipeline(file_path)
            
            if result['success']:
                print("✅ Pipeline successful!\n")
                
                # Show extraction
                print("📝 EXTRACTED TEXT:")
                print("-" * 80)
                text = result['extraction'].get('text', '')
                if len(text) > 300:
                    print(text[:300] + "\n... [truncated]")
                else:
                    print(text)
                print("-" * 80)
                
                # Show prediction
                print("\n💊 PREDICTION RESULT:")
                print("-" * 80)
                pred = result.get('prediction', {})
                medicine = pred.get('predicted_medicine', 'Unknown')
                confidence = pred.get('confidence', 0)
                date_str = pred.get('date', 'N/A')
                summary = pred.get('summary', '')
                
                print(f"Medicine:    {medicine}")
                # print(f"Confidence:  {confidence:.1%}" if isinstance(confidence, (int, float)) else f"Confidence:  {confidence}")
                print(f"Date:        {date_str}")
                print(f"Summary:     {summary[:80]}...")
                print("-" * 80)
                
                # Show output file
                print(f"\n💾 Saved to: {result.get('output_file', 'N/A')}")
                
        elif option == "3":
            # Batch prediction
            print("🔍 Extracting text & generating prediction...")
            result = client.pipeline(file_path)
            
            if result['success']:
                print("✅ Processing successful!\n")
                
                pred = result.get('prediction', {})
                output_data = {
                    'date': pred.get('date', 'N/A'),
                    'summary': pred.get('summary', ''),
                    'medicine_recommendation': pred.get('predicted_medicine', 'Unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                print("📊 OUTPUT DATA:")
                print("-" * 80)
                for key, value in output_data.items():
                    if key != 'timestamp':
                        print(f"{key.replace('_', ' ').title()}: {value}")
                print("-" * 80)
                print(f"\n💾 Full data saved to: {result.get('output_file', 'N/A')}")
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE!")
        print("="*80 + "\n")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API server")
        print("   Make sure to run: python api.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def show_menu():
    """Display processing options"""
    print("\n" + "="*80)
    print("SELECT PROCESSING OPTION:")
    print("="*80)
    print("\n  1️⃣  Extract Text Only")
    print("  2️⃣  Extract & Predict (Complete Pipeline)")
    print("  3️⃣  View Structured Output (Date - Summary - Medicine)")
    print("\n" + "="*80)
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("❌ Invalid choice. Please enter 1, 2, or 3")


def open_output_file(file_path):
    """Ask if user wants to open output file"""
    if not os.path.exists(file_path):
        return
    
    try:
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{file_path}"')
        else:
            os.system(f'xdg-open "{file_path}"')
    except Exception as e:
        print(f"Could not open file: {e}")


def main():
    """Main function"""
    show_welcome()
    
    # Pick file
    file_path = pick_file()
    
    # Validate
    valid, message = validate_file(file_path)
    if not valid:
        print(f"❌ {message}")
        return
    
    # Show menu and process
    option = show_menu()
    success = process_file(file_path, option)
    
    if success:
        ask_open = input("\n📂 Open output file? (y/n): ").strip().lower()
        if ask_open == 'y':
            # Try to find and open the latest output file
            output_dir = "output"
            if os.path.exists(output_dir):
                files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    open_output_file(latest_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
