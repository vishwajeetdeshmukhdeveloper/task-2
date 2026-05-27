# Medical Document Processing System - Complete Architecture Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Detailed Code Explanations](#detailed-code-explanations)
6. [How to Use](#how-to-use)
7. [API Reference](#api-reference)

---

## System Overview

This is a comprehensive **Medical Document Processing & Prediction System** that:
- Extracts text from PDFs and medical report images using OCR
- Cleans and processes medical findings
- Generates medicine recommendations using ML
- Provides REST API for integration
- Supports batch processing of multiple documents
- Outputs organized reports (Date, Summary, Recommendation)

**Key Technologies:**
- **pdfplumber** - PDF text extraction
- **pytesseract** - Image OCR
- **scikit-learn** - ML predictions (Naive Bayes + TF-IDF)
- **Flask** - REST API server
- **tkinter** - GUI file picker
- **Pillow** - Image processing

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  upload_pdf.py   │  │  API Client      │  │  process_all_    │
│  │  (GUI File       │  │  (Python Lib)    │  │  reports.py      │
│  │   Picker)        │  │                  │  │  (Batch Mode)    │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
│           │                     │                      │
└───────────┼─────────────────────┼──────────────────────┼──────────┘
            │                     │                      │
            └─────────────────────┼──────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   FLASK REST API (api.py)  │
                    │   localhost:5000           │
                    └─────────────┬──────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  TEXT EXTRACTION │  │  TEXT CLEANING   │  │  ML PREDICTION   │
│  Layer           │  │  Layer           │  │  Layer           │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ text_extraction  │  │ medical_text_    │  │ inference_model  │
│ .py              │  │ cleaner.py       │  │ .py              │
│                  │  │                  │  │                  │
│ • extract_pdf()  │  │ • clean_summary()│  │ • predict_single()
│ • extract_image()│  │ • regex patterns │  │ • predict_batch()
│ • extract_file() │  │ • find findings  │  │ • save to file   │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  ML MODEL (ml/)     │
                    ├─────────────────────┤
                    │ • Trained weights   │
                    │ • TF-IDF vectorizer │
                    │ • 77.27% accuracy   │
                    │ • 24 training docs  │
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  OUTPUT LAYER       │
                    ├─────────────────────┤
                    │ • Organized reports │
                    │ • Date/Summary/Rec  │
                    │ • Text files        │
                    │ • output/ folder    │
                    └─────────────────────┘
```

---

## Core Components

### 1. **Text Extraction Layer** (`text_extraction.py`)
Handles file type detection and unified text extraction interface.

```python
# Purpose: Unified interface for extracting text from PDFs and images

class TextExtractor:
    """Polymorphic text extractor with auto file-type detection"""
    
    def extract_file(self, file_path: str) -> dict:
        """
        AUTO-DETECTS file type and extracts accordingly
        - .pdf → uses pdfplumber
        - .jpg/.png/.bmp → uses pytesseract OCR
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return self.extract_image(file_path)
    
    def extract_pdf(self, pdf_path: str) -> dict:
        """
        Uses pdfplumber library to:
        1. Open PDF file
        2. Extract text from all pages
        3. Extract tables if present
        4. Return combined text
        """
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                # Extract page text
                text += page.extract_text() + "\n"
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    text += str(tables) + "\n"
        
        return {"text": text, "file": pdf_path, "type": "pdf"}
    
    def extract_image(self, image_path: str) -> dict:
        """
        Uses pytesseract OCR to:
        1. Open image with Pillow
        2. Convert to grayscale for better OCR
        3. Use Tesseract engine for character recognition
        4. Return extracted text
        """
        from PIL import Image
        import pytesseract
        
        # Grayscale preprocessing improves OCR accuracy
        image = Image.open(image_path).convert('L')
        
        # Tesseract recognizes text in image
        text = pytesseract.image_to_string(image)
        
        return {"text": text, "file": image_path, "type": "image"}
```

### 2. **Medical Text Cleaner** (`medical_text_cleaner.py`)
Processes raw extraction to extract key medical findings.

```python
# Purpose: Clean verbose medical text and extract key findings

class MedicalTextCleaner:
    """
    Cleans raw extracted text and finds medical patterns
    Input: Raw PDF/Image text (verbose, noisy)
    Output: Clean summary + recommendation tuple
    """
    
    def clean_summary(self, raw_text: str) -> tuple:
        """
        Returns: (cleaned_summary, recommendation)
        
        Process:
        1. Extract key findings using regex patterns
        2. Remove page markers, metadata, noise
        3. Generate recommendation based on findings
        """
        # Step 1: Find important medical keywords
        findings = self.extract_key_findings(raw_text)
        
        # Step 2: Create readable summary (1-3 sentences)
        cleaned_summary = self._create_summary(findings)
        
        # Step 3: Generate specific recommendation
        recommendation = self.get_recommended_action(findings)
        
        return (cleaned_summary, recommendation)
    
    def extract_key_findings(self, text: str) -> dict:
        """
        Uses regex patterns to find:
        - Glucose levels
        - Blood pressure
        - Cholesterol
        - Other medical indicators
        """
        import re
        findings = {}
        
        # Look for glucose pattern: "Glucose: XXX mg/dL"
        glucose_match = re.search(r'Glucose[:\s]+(\d+)\s*mg', text)
        if glucose_match:
            findings['glucose'] = int(glucose_match.group(1))
        
        # Look for blood pressure pattern: "BP: XXX/YYY"
        bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', text)
        if bp_match:
            findings['bp'] = (int(bp_match.group(1)), int(bp_match.group(2)))
        
        # Similar patterns for other medical markers...
        
        return findings
    
    def get_recommended_action(self, findings: dict) -> str:
        """
        Based on extracted findings, generate specific recommendations:
        - High glucose → "Manage glucose levels"
        - High cholesterol → "Reduce fat intake"
        - Low hemoglobin → "Iron supplement"
        """
        recommendations = []
        
        if findings.get('glucose', 0) > 126:
            recommendations.append("Manage glucose levels with diet and exercise")
        
        if findings.get('cholesterol', 0) > 200:
            recommendations.append("Reduce saturated fat intake")
        
        return " ".join(recommendations) or "Follow regular checkups"
```

### 3. **ML Prediction Model** (`inference_model.py`)
Generates medicine recommendations using trained model.

```python
# Purpose: Make medical predictions from cleaned text

class MedicalPredictor:
    """
    Uses trained Naive Bayes model to predict medicine recommendations
    """
    
    def __init__(self):
        """
        Load pre-trained model components:
        1. Naive Bayes classifier
        2. TF-IDF vectorizer (converts text to numbers)
        """
        import pickle
        
        # Load model file
        with open('models/medical_model_v1.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer (TF-IDF converts words to features)
        with open('models/medical_model_v1_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def predict_single(self, medical_summary: str) -> dict:
        """
        Make prediction for single medical summary
        
        Process:
        1. Convert text to TF-IDF features (500-dimensional vector)
        2. Feed to Naive Bayes model
        3. Get prediction + confidence score
        """
        # Step 1: Convert text to features
        # TF-IDF: Term Frequency-Inverse Document Frequency
        # Assigns importance to words across all documents
        features = self.vectorizer.transform([medical_summary])
        
        # Step 2: Get prediction
        prediction = self.model.predict(features)[0]
        
        # Step 3: Get confidence (probability of this class)
        confidence = self.model.predict_proba(features).max()
        
        return {
            "predicted_medicine": prediction,
            "confidence": confidence,
            "summary": medical_summary
        }
    
    def save_prediction_to_file(self, date, clean_summary, medicine, output_file):
        """
        Save organized prediction to text file
        
        Format:
        ====================================
        MEDICAL PREDICTION REPORT
        ====================================
        
        Date: 2026-05-17
        
        Summary:
        [medical findings]
        
        Recommendation:
        [medicine/action]
        
        ====================================
        """
        with open(output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MEDICAL PREDICTION REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Date: {date}\n")
            f.write(f"\nSummary:\n{clean_summary}\n")
            f.write(f"\nRecommendation:\n{medicine}\n\n")
            f.write("=" * 100 + "\n")
```

### 4. **REST API** (`api.py`)
Flask server providing 8 HTTP endpoints.

```python
# Purpose: Expose all system functions via REST API

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint: GET /health
    Purpose: Check if API is running
    Response: {"status": "healthy"}
    """
    return jsonify({"status": "healthy"})

@app.route('/api/extract', methods=['POST'])
def extract():
    """
    Endpoint: POST /api/extract
    Purpose: Extract text from uploaded file
    
    Request:
    - file: PDF or image file
    
    Process:
    1. Receive file upload
    2. Create TextExtractor instance
    3. Auto-detect file type
    4. Extract text
    5. Return extracted content
    """
    file = request.files['file']
    
    # Save uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    
    # Extract text
    extractor = TextExtractor()
    result = extractor.extract_file(file_path)
    
    return jsonify({
        "success": True,
        "text": result['text'],
        "file_type": result['type']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint: POST /api/predict
    Purpose: Generate medicine prediction from text
    
    Request JSON:
    {
        "summary": "Patient has high glucose levels"
    }
    
    Process:
    1. Receive medical summary
    2. Load ML model
    3. Convert text to features
    4. Get prediction
    5. Return recommendation
    """
    data = request.get_json()
    summary = data['summary']
    
    # Create predictor and make prediction
    predictor = MedicalPredictor()
    result = predictor.predict_single(summary)
    
    return jsonify({
        "success": True,
        "predicted_medicine": result['predicted_medicine'],
        "confidence": result['confidence']
    })

@app.route('/api/pipeline', methods=['POST'])
def pipeline():
    """
    Endpoint: POST /api/pipeline
    Purpose: Complete workflow: Extract → Clean → Predict → Save
    
    Request:
    - file: PDF or image
    
    Complete Process:
    1. Extract text from file
    2. Clean summary (remove noise)
    3. Predict medicine
    4. Save to output file
    5. Return results
    """
    file = request.files['file']
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    
    # Step 1: Extract
    extractor = TextExtractor()
    extracted = extractor.extract_file(file_path)
    text = extracted['text']
    
    # Step 2: Clean
    cleaner = MedicalTextCleaner()
    clean_summary, recommendation = cleaner.clean_summary(text)
    
    # Step 3: Predict
    predictor = MedicalPredictor()
    prediction = predictor.predict_single(clean_summary)
    
    # Step 4: Save
    from datetime import datetime
    date = datetime.now().strftime('%Y-%m-%d')
    output_file = f"output/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    predictor.save_prediction_to_file(
        date, 
        clean_summary, 
        prediction['predicted_medicine'], 
        output_file
    )
    
    # Step 5: Return results
    return jsonify({
        "success": True,
        "extracted_text": text,
        "clean_summary": clean_summary,
        "predicted_medicine": prediction['predicted_medicine'],
        "confidence": prediction['confidence'],
        "output_file": output_file
    })
```

### 5. **File Picker** (`upload_pdf.py`)
Interactive GUI for file selection and processing.

```python
# Purpose: User-friendly GUI for file selection and processing

def show_welcome():
    """Display welcome message"""
    print("\n" + "="*80)
    print("MEDICAL DOCUMENT PROCESSOR")
    print("="*80 + "\n")

def pick_file():
    """
    Open file picker dialog to select file
    
    Uses tkinter for GUI:
    1. Create root window
    2. Hide it (we only want the dialog)
    3. Show file picker
    4. Return selected file path
    """
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    # File filter to show only PDFs/images
    filetypes = (
        ('PDF Files', '*.pdf'),
        ('Image Files', '*.jpg *.png *.bmp'),
        ('All Files', '*.*')
    )
    
    # Show dialog and get file path
    file_path = filedialog.askopenfilename(
        title='Select a Medical Report',
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path

def process_file(file_path, mode):
    """
    Process selected file in one of 3 modes:
    
    Mode 1: Extract Only
    - Extract text and display preview
    
    Mode 2: Extract + Predict
    - Extract → Clean → Predict → Display results
    
    Mode 3: View Structured Output
    - Show formatted Date/Summary/Recommendation
    """
    if mode == 1:
        # Mode 1: Extract only
        extractor = TextExtractor()
        result = extractor.extract_file(file_path)
        print("\nExtracted Text:")
        print(result['text'][:500] + "...")
    
    elif mode == 2:
        # Mode 2: Extract + Predict
        client = MedicalAPIClient()
        result = client.pipeline(file_path)
        
        print("\n[EXTRACTION]")
        print(result['extracted_text'][:300] + "...\n")
        
        print("[SUMMARY]")
        print(result['clean_summary'] + "\n")
        
        print("[RECOMMENDATION]")
        print(result['predicted_medicine'] + "\n")
    
    elif mode == 3:
        # Mode 3: View structured output
        # Read the saved file and display
        files = sorted(Path("output").glob("pipeline_*.txt"), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                print("\n" + f.read())
```

### 6. **Image File Picker** (`extractors/image_extractor.py`)
Enhanced file picker for image selection.

```python
# Purpose: File picker specifically for image files with fallback modes

def pick_image_file():
    """
    Try GUI first, fall back to terminal menu if needed
    
    GUI Mode:
    1. Create tkinter window
    2. Show file dialog
    3. User selects file
    4. Return path
    
    Fallback Terminal Mode:
    If GUI unavailable (no display/headless):
    1. Search for image files
    2. Show numbered menu
    3. User enters selection number
    4. Return selected path
    """
    try:
        # Try GUI mode first
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title='Select an Image File',
            filetypes=[('Image Files', '*.jpg *.png *.bmp')]
        )
        
        root.destroy()
        return file_path if file_path else None
    
    except Exception as e:
        # If GUI fails, use terminal menu
        return pick_image_file_terminal()

def pick_image_file_terminal():
    """
    Terminal-based file picker:
    1. Find all image files in directory
    2. Display numbered list
    3. Get user input
    4. Return selected path
    """
    import os
    
    # Find all images
    image_files = []
    for root_dir, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.jpg', '.png', '.bmp')):
                full_path = os.path.join(root_dir, file)
                image_files.append(full_path)
    
    # Display menu
    print("\nFound image files:\n")
    for idx, file in enumerate(image_files, 1):
        print(f"  [{idx}] {file}")
    
    print(f"\n  [0] Enter path manually")
    print(f"  [C] Cancel\n")
    
    # Get user choice
    choice = input("Select an image (enter number): ")
    
    if choice == 'c':
        return None
    
    if choice == '0':
        return input("Enter image file path: ")
    
    try:
        idx = int(choice) - 1
        return image_files[idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return None
```

### 7. **Batch Processing** (`process_all_reports.py`)
Process all PDFs in directory at once.

```python
# Purpose: Batch process entire directory of medical reports

def main():
    """
    Batch processing workflow:
    1. Find all PDF/image files
    2. Process each file
    3. Generate combined report
    """
    
    # Step 1: Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Step 2: Process all files in samples/pdf/
    print("Processing all medical reports...\n")
    
    results = orchestrator.run_batch_pipeline(
        input_dir="samples/pdf/",
        output_dir="output/"
    )
    
    # Step 3: Generate batch report
    batch_file = f"output/pipeline_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(batch_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("BATCH PROCESSING REPORT\n")
        f.write("="*100 + "\n\n")
        
        # Write each result
        for idx, result in enumerate(results, 1):
            f.write(f"[{idx}] {result['file']}\n")
            f.write(f"Medicine: {result['prediction']}\n")
            f.write("-"*100 + "\n\n")
    
    print(f"\nBatch report saved: {batch_file}")
```

### 8. **Pipeline Orchestrator** (`pipeline.py`)
Coordinates entire workflow.

```python
# Purpose: Orchestrate complete extract→predict→save workflow

class PipelineOrchestrator:
    """
    Manages the complete pipeline:
    Extract → Clean → Predict → Train → Save
    """
    
    def run_batch_pipeline(self, input_dir, output_dir):
        """
        Process entire directory:
        
        For each file:
        1. Extract text
        2. Clean summary
        3. Make prediction
        4. Save to file
        """
        results = []
        
        # Find all PDFs and images
        files = list(Path(input_dir).glob("*"))
        
        for file_path in files:
            print(f"Processing: {file_path.name}")
            
            # Step 1: Extract
            extractor = TextExtractor()
            extraction = extractor.extract_file(str(file_path))
            text = extraction['text']
            
            # Step 2: Clean
            cleaner = MedicalTextCleaner()
            clean_summary, recommendation = cleaner.clean_summary(text)
            
            # Step 3: Predict
            predictor = MedicalPredictor()
            prediction = predictor.predict_single(clean_summary)
            
            # Step 4: Save
            date = datetime.now().strftime('%Y-%m-%d')
            predictor.save_prediction_to_file(
                date,
                clean_summary,
                prediction['predicted_medicine'],
                f"{output_dir}/{file_path.stem}_prediction.txt"
            )
            
            results.append({
                "file": file_path.name,
                "prediction": prediction['predicted_medicine'],
                "confidence": prediction['confidence']
            })
        
        return results
```

---

## Data Flow

### Flow 1: Single File Processing (API Pipeline)
```
User Upload File (PDF/Image)
         ↓
TextExtractor.extract_file()
    (Auto-detect type)
         ↓
extract_pdf() OR extract_image()
    (Get raw text)
         ↓
MedicalTextCleaner.clean_summary()
    (Remove noise, extract findings)
         ↓
MedicalPredictor.predict_single()
    (TF-IDF → Naive Bayes → Prediction)
         ↓
save_prediction_to_file()
    (Write organized output)
         ↓
Return JSON Response
```

### Flow 2: Batch Directory Processing
```
process_all_reports.py
         ↓
Find all files in samples/pdf/
         ↓
For each file:
    TextExtractor → Clean → Predict → Save
         ↓
Generate combined batch report
         ↓
output/pipeline_batch_TIMESTAMP.txt
```

### Flow 3: Interactive GUI (upload_pdf.py)
```
Start upload_pdf.py
         ↓
Show welcome message
         ↓
pick_file() → User selects file
         ↓
Show mode menu (1/2/3)
         ↓
Mode 1: Extract only
Mode 2: Full pipeline
Mode 3: View formatted output
         ↓
Display results
```

---

## Detailed Code Explanations

### Text Extraction Process

**PDF Extraction (pdfplumber):**
```python
# Line-by-line: extract_pdf()

import pdfplumber

def extract_pdf(pdf_path):
    # Step 1: Open PDF file
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        
        # Step 2: Iterate through each page
        for page in pdf.pages:
            
            # Step 3: Extract text from page
            # pdfplumber's extract_text() handles:
            # - Text location analysis
            # - Font recognition
            # - Word positioning
            page_text = page.extract_text()
            text += page_text + "\n"
            
            # Step 4: Extract tables if present
            # Finds structured table data
            tables = page.extract_tables()
            if tables:
                # Convert table to string format
                text += str(tables) + "\n"
    
    # Step 5: Return structured result
    return {
        "text": text,
        "file": pdf_path,
        "type": "pdf"
    }
```

**Image Extraction (pytesseract):**
```python
# Line-by-line: extract_image()

from PIL import Image
import pytesseract

def extract_image(image_path):
    # Step 1: Open image file
    image = Image.open(image_path)
    
    # Step 2: Convert to grayscale
    # Why? Grayscale improves OCR accuracy by:
    # - Removing color noise
    # - Enhancing character contrast
    # - Reducing processing complexity
    image = image.convert('L')
    
    # Step 3: Pass to pytesseract
    # pytesseract uses Google's Tesseract engine:
    # - Recognizes characters in image
    # - Handles multiple languages
    # - Returns text string
    extracted_text = pytesseract.image_to_string(
        image,
        lang='eng',        # English language
        config='--psm 3'   # PSM 3 = auto page segmentation
    )
    
    # Step 4: Clean up whitespace
    text = extracted_text.strip()
    
    # Step 5: Return result
    return {
        "text": text,
        "file": image_path,
        "type": "image"
    }
```

### Medical Text Cleaning Process

```python
# Line-by-line: clean_summary()

def clean_summary(self, raw_text):
    # Step 1: Extract key findings using regex
    findings = self.extract_key_findings(raw_text)
    
    # Step 2: Build readable summary
    # Only includes relevant medical information
    cleaned_summary = self._create_summary(findings)
    
    # Step 3: Generate recommendation based on findings
    # Example: glucose > 126 → "Manage glucose"
    recommendation = self.get_recommended_action(findings)
    
    # Step 4: Return tuple (summary, recommendation)
    return (cleaned_summary, recommendation)

def extract_key_findings(self, text):
    # Step 1: Import regex library
    import re
    findings = {}
    
    # Step 2: Search for glucose pattern
    # Pattern: "Glucose" followed by number and "mg"
    glucose_match = re.search(r'Glucose[:\s]+(\d+)\s*mg', text)
    if glucose_match:
        # Extract the number and convert to int
        findings['glucose'] = int(glucose_match.group(1))
    
    # Step 3: Similar patterns for other markers
    # Blood pressure pattern
    bp_match = re.search(r'BP[:\s]+(\d+)/(\d+)', text)
    if bp_match:
        findings['bp'] = (int(bp_match.group(1)), int(bp_match.group(2)))
    
    # Cholesterol pattern
    chol_match = re.search(r'Cholesterol[:\s]+(\d+)', text)
    if chol_match:
        findings['cholesterol'] = int(chol_match.group(1))
    
    # Step 4: Return dictionary of findings
    return findings
```

### ML Prediction Process

```python
# Line-by-line: predict_single()

def predict_single(self, medical_summary):
    # Step 1: Import numpy for array operations
    import numpy as np
    
    # Step 2: Convert text to TF-IDF features
    # TF-IDF = Term Frequency - Inverse Document Frequency
    # 
    # TF (Term Frequency):
    # - How often a word appears in this document
    # - Measures word importance in current text
    # 
    # IDF (Inverse Document Frequency):
    # - Log(total docs / docs with this word)
    # - Words in all docs get low weight
    # - Rare words get high weight
    # 
    # Result: 500-dimensional vector (500 features)
    features = self.vectorizer.transform([medical_summary])
    
    # Step 3: Make prediction using Naive Bayes
    # Naive Bayes uses probability:
    # P(Medicine | Features) = P(Features | Medicine) × P(Medicine) / P(Features)
    # 
    # Assumes features are independent (naive assumption)
    # Fast and effective for text classification
    prediction = self.model.predict(features)[0]
    
    # Step 4: Get confidence score
    # Get probability for each class
    probabilities = self.model.predict_proba(features)[0]
    
    # Take maximum probability as confidence
    confidence = probabilities.max()
    
    # Step 5: Return result dictionary
    return {
        "predicted_medicine": prediction,      # Which medicine to recommend
        "confidence": confidence,               # How confident (0-1 scale)
        "summary": medical_summary
    }
```

### ML Model Training (One-time process)

```python
# Line-by-line: train_model()

def train_model(training_data):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    
    # Step 1: Extract training texts and labels
    # training_data = [
    #   {"text": "glucose 150mg", "medicine": "Metformin"},
    #   {"text": "high cholesterol", "medicine": "Statin"},
    #   ...24 total records
    # ]
    texts = [d['text'] for d in training_data]
    labels = [d['medicine'] for d in training_data]
    
    # Step 2: Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,        # Use top 500 words
        ngram_range=(1, 2),      # Consider 1-2 word combinations
        min_df=2,                # Word must appear in ≥2 documents
        max_df=0.8               # Word can't appear in >80% of documents
    )
    
    # Step 3: Fit vectorizer on training texts
    # Vectorizer learns vocabulary and IDF weights
    features = vectorizer.fit_transform(texts)
    
    # Step 4: Create Naive Bayes classifier
    model = MultinomialNB()
    
    # Step 5: Train model on features and labels
    # Model learns probability distribution of each medicine
    model.fit(features, labels)
    
    # Step 6: Save trained model
    with open('models/medical_model_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Step 7: Save vectorizer
    # Vectorizer must be saved because it's needed at prediction time
    with open('models/medical_model_v1_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Step 8: Calculate and print accuracy
    predictions = model.predict(features)
    accuracy = (predictions == labels).mean()
    print(f"Model Accuracy: {accuracy:.2%}")  # 77.27%
```

### Output File Saving

```python
# Line-by-line: save_prediction_to_file()

def save_prediction_to_file(self, date, clean_summary, medicine, output_file):
    # Step 1: Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        
        # Step 2: Write report header (decorative)
        f.write("=" * 100 + "\n")
        f.write("MEDICAL PREDICTION REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Step 3: Write date
        # Format: "Date: 2026-05-17"
        f.write(f"Date: {date}\n")
        
        # Step 4: Write section header and summary
        f.write(f"\nSummary:\n")
        f.write(f"{clean_summary}\n")
        
        # Step 5: Write recommendation
        # Format: "Recommendation: [medicine name]"
        f.write(f"\nRecommendation:\n")
        f.write(f"{medicine}\n\n")
        
        # Step 6: Write footer
        f.write("=" * 100 + "\n")
    
    # Result: Clean, organized text file with:
    # ✓ Date
    # ✓ Summary
    # ✓ Recommendation
    # ✓ No confidence scores
    # ✓ No timestamps
```

---

## How to Use

### 1. **Interactive GUI Mode (Recommended for Single Files)**
```bash
python upload_pdf.py
```
- Launches welcome screen
- Shows file picker
- Displays processing modes
- Shows results

### 2. **Batch Process All Files**
```bash
python process_all_reports.py
```
- Processes all PDFs in samples/pdf/
- Generates combined report
- Saves to output/pipeline_batch_TIMESTAMP.txt

### 3. **Start REST API Server**
```bash
python api.py
```
- Starts Flask server on localhost:5000
- Available endpoints documented at /api/docs
- Can be called by external applications

### 4. **Use Python API Client**
```python
from api_client import MedicalAPIClient

client = MedicalAPIClient()

# Extract text
result = client.extract('path/to/file.pdf')
print(result['text'])

# Make prediction
result = client.predict('High glucose levels')
print(result['predicted_medicine'])

# Full pipeline
result = client.pipeline('path/to/file.pdf')
print(result['output_file'])
```

### 5. **Image File Picker**
```python
from extractors.image_extractor import pick_image_file, extract_image

# Pick image file (GUI or terminal menu)
image_path = pick_image_file()

if image_path:
    # Extract text
    text = extract_image(image_path)
    print(text)
```

---

## API Reference

### Health Check
```
GET /health
→ Returns: {"status": "healthy"}
```

### Extract Text
```
POST /api/extract
Content-Type: multipart/form-data
- file: [PDF or image file]

→ Returns: {
    "success": true,
    "text": "extracted text...",
    "file_type": "pdf"
  }
```

### Make Prediction
```
POST /api/predict
Content-Type: application/json
{
    "summary": "Patient has high glucose levels"
}

→ Returns: {
    "success": true,
    "predicted_medicine": "Metformin",
    "confidence": 0.85
  }
```

### Complete Pipeline
```
POST /api/pipeline
Content-Type: multipart/form-data
- file: [PDF or image file]

→ Returns: {
    "success": true,
    "extracted_text": "...",
    "clean_summary": "...",
    "predicted_medicine": "Metformin",
    "confidence": 0.85,
    "output_file": "output/pipeline_20260517_141813.txt"
  }
```

### Batch Predictions
```
POST /api/batch-predict
Content-Type: application/json
{
    "summaries": ["summary1", "summary2"]
}

→ Returns: {
    "success": true,
    "predictions": [
        {"medicine": "Metformin", "confidence": 0.85},
        {"medicine": "Aspirin", "confidence": 0.92}
    ]
  }
```

### Model Information
```
GET /api/model-info
→ Returns: {
    "model_type": "naive_bayes",
    "accuracy": 0.7727,
    "training_samples": 24,
    "features": 500
  }
```

### Train Model
```
POST /api/train
Content-Type: application/json

→ Returns: {
    "success": true,
    "accuracy": 0.7727,
    "message": "Model retrained successfully"
  }
```

---

## System Configuration

**Key Libraries & Versions:**
- pdfplumber 0.10.3 - PDF extraction
- pytesseract 0.3.10 - Image OCR
- scikit-learn 1.3.2 - ML model
- Flask 2.3.3 - REST API
- Pillow 10.1.0 - Image processing
- tkinter (built-in) - GUI

**Model Configuration:**
- Type: Naive Bayes Classifier
- Vectorizer: TF-IDF (500 features, bigrams)
- Training Data: 24 medical records
- Accuracy: 77.27%
- Supported Languages: English

**Output Format:**
- Each prediction saved as organized text file
- Location: output/ directory
- Format: Date / Summary / Recommendation (separate lines)
- No confidence scores in output
- UTF-8 encoding for all files

---

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│ USER INTERFACES                              │
│ • upload_pdf.py (GUI File Picker)           │
│ • image_extractor.py (Image Picker)         │
│ • API (HTTP Endpoints)                      │
│ • process_all_reports.py (Batch)            │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ EXTRACTION LAYER                             │
│ • text_extraction.py (PDF + Image)          │
│ • Auto file-type detection                  │
│ • Unified interface                         │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ CLEANING LAYER                               │
│ • medical_text_cleaner.py                   │
│ • Regex-based finding extraction            │
│ • Summary generation                        │
│ • Recommendation logic                      │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ PREDICTION LAYER                             │
│ • inference_model.py (ML predictions)       │
│ • TF-IDF vectorizer                         │
│ • Naive Bayes classifier                    │
│ • 77.27% accuracy                           │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ OUTPUT LAYER                                 │
│ • Organized text files                      │
│ • Date / Summary / Recommendation           │
│ • Batch report generation                   │
│ • output/ directory storage                 │
└─────────────────────────────────────────────┘
```
# Thank You
This completes a full circle of medical document processing from upload to final organized report!
