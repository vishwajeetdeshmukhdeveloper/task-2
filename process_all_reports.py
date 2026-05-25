#!/usr/bin/env python
"""
Batch process all medical reports at once
Extracts, predicts, and saves results for all PDFs in a directory
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from pipeline import PipelineOrchestrator


def main():
    """Run batch processing on all reports"""
    
    # Input directory with PDFs
    input_dir = 'samples/pdf'
    
    # Verify directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Directory '{input_dir}' not found")
        return False
    
    # Get all PDFs
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ Error: No PDF files found in '{input_dir}'")
        return False
    
    print("=" * 100)
    print("BATCH MEDICAL REPORT PROCESSING")
    print("=" * 100)
    print(f"\n[INPUT] Directory: {input_dir}")
    print(f"[FOUND] {len(pdf_files)} PDF files:")
    
    for i, pdf in enumerate(sorted(pdf_files), 1):
        print(f"   {i:2d}. {pdf.name}")
    
    print("\n" + "=" * 100)
    print("Starting batch processing...")
    print("=" * 100 + "\n")
    
    # Initialize pipeline
    orchestrator = PipelineOrchestrator()
    
    # Run batch pipeline
    results = orchestrator.run_batch_pipeline(
        input_directory=input_dir,
        auto_train=True  # Retrain model after processing
    )
    
    # Print summary
    print("\n" + "=" * 100)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 100)
    print(f"\n[COMPLETE] Processed: {results.get('total_files', 0)} reports")
    print(f"[EXTRACT] Extracted: {results.get('extracted_files', 0)} files")
    print(f"[PREDICT] Predictions: {results.get('predictions_made', 0)} predictions")
    print(f"[SAVED] Output Files: {results.get('saved_files', 0)} files")
    
    if results.get('saved_files_list'):
        print(f"\n[OUTPUT] Files:")
        for file in results['saved_files_list'][:10]:  # Show first 10
            print(f"   [OK] {Path(file).name}")
        if len(results['saved_files_list']) > 10:
            print(f"   ... and {len(results['saved_files_list']) - 10} more")
    
    if results.get('model_trained'):
        print(f"\n[MODEL] Training:")
        print(f"   [DONE] Model retrained with {results.get('training_samples', 0)} samples")
        print(f"   [ACCURACY] New accuracy: {results.get('training_accuracy', 0):.2%}")
    
    print(f"\n[REPORT] Summary: {results.get('summary_file', 'N/A')}")
    print("\n" + "=" * 100 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
