"""
Pipeline Orchestrator
Manages complete workflow: Extract -> Predict -> Train -> Save
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from text_extraction import TextExtractor
from inference_model import MedicalPredictor
from ml.data_processor import MedicalDataProcessor
from ml.model import MedicalPredictionModel


class PipelineOrchestrator:
    """Orchestrates the complete medical data pipeline"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration dictionary with paths and settings
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.extractor = TextExtractor(output_dir=self.config['output_dir'])
        self.predictor = MedicalPredictor()
        self.processor = MedicalDataProcessor()
        self.model = MedicalPredictionModel()
        
        # Create directories
        for dir_path in [self.config['output_dir'], self.config['models_dir'], self.config['data_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Pipeline log
        self.log = {
            'started': datetime.now().isoformat(),
            'steps': [],
            'errors': []
        }
    
    def _default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'output_dir': 'output',
            'models_dir': 'models',
            'data_dir': 'data',
            'samples_dir': 'samples',
            'auto_train': True,
            'save_predictions': True,
            'training_model_type': 'naive_bayes'
        }
    
    def log_step(self, step_name: str, status: str, details: Dict = None):
        """Log a pipeline step"""
        log_entry = {
            'step': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.log['steps'].append(log_entry)
        print(f"✓ [{step_name}] {status}")
    
    def log_error(self, error: str, details: Dict = None):
        """Log an error"""
        error_entry = {
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.log['errors'].append(error_entry)
        print(f"✗ ERROR: {error}")
    
    # ============ STEP 1: EXTRACT ============
    
    def extract_from_file(self, file_path: str) -> Dict:
        """Extract text from a single file"""
        print(f"\n{'='*100}")
        print(f"STEP 1: EXTRACT FROM FILE")
        print(f"{'='*100}")
        
        if not os.path.exists(file_path):
            self.log_error(f"File not found: {file_path}")
            return {'success': False, 'error': 'File not found'}
        
        print(f"📄 Extracting from: {file_path}")
        
        # Extract
        extraction = self.extractor.extract_file(file_path)
        
        if extraction['success']:
            # Save extraction
            saved_path = self.extractor.save_extraction(extraction)
            self.log_step('extract_file', 'success', {
                'file': file_path,
                'file_type': extraction['file_type'],
                'text_length': len(extraction['text']),
                'saved_to': saved_path
            })
            
            print(f"✓ Extracted {len(extraction['text'])} characters")
            print(f"✓ Saved to: {saved_path}")
            
            return extraction
        else:
            self.log_error(f"Extraction failed: {extraction.get('error')}")
            return extraction
    
    def extract_from_directory(self, directory: str, recursive: bool = False) -> List[Dict]:
        """Extract text from all files in a directory"""
        print(f"\n{'='*100}")
        print(f"STEP 1: EXTRACT FROM DIRECTORY")
        print(f"{'='*100}")
        print(f"📁 Extracting from: {directory}")
        
        # Extract all files
        extractions = self.extractor.extract_from_directory(directory, recursive=recursive)
        
        # Save all extractions
        saved_files = []
        for extraction in extractions:
            if extraction['success']:
                saved_path = self.extractor.save_extraction(extraction)
                saved_files.append({
                    'file': extraction['file_path'],
                    'saved_to': saved_path,
                    'text_length': len(extraction['text'])
                })
        
        self.log_step('extract_directory', 'success', {
            'directory': directory,
            'files_processed': len(extractions),
            'files_saved': len(saved_files)
        })
        
        print(f"✓ Processed {len(extractions)} files")
        print(f"✓ Saved {len(saved_files)} extracted files")
        
        return extractions
    
    # ============ STEP 2: PREDICT ============
    
    def predict_from_text(self, text: str, date: str = None) -> Dict:
        """Generate prediction from extracted text"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n  📊 Generating prediction...")
        
        # Generate prediction using model.predict directly
        prediction = self.predictor.model.predict(text)
        
        self.log_step('predict', 'success', {
            'text_length': len(text),
            'date': date,
            'prediction': prediction.get('predicted_medicine', 'N/A'),
            'confidence': prediction.get('confidence', 'N/A')
        })
        
        print(f"  ✓ Prediction: {prediction.get('predicted_medicine', 'N/A')}")
        print(f"  ✓ Confidence: {prediction.get('confidence', 0):.2%}")
        
        return prediction
    
    def predict_batch(self, texts: List[str], dates: List[str] = None) -> List[Dict]:
        """Generate predictions for multiple texts"""
        print(f"\n{'='*100}")
        print(f"STEP 2: BATCH PREDICTIONS")
        print(f"{'='*100}")
        print(f"📊 Generating {len(texts)} predictions...")
        
        if dates is None:
            dates = [datetime.now().strftime('%Y-%m-%d')] * len(texts)
        
        predictions = []
        for i, (text, date) in enumerate(zip(texts, dates), 1):
            print(f"  [{i}/{len(texts)}]", end=" ")
            pred = self.predict_from_text(text, date)
            predictions.append({
                'date': date,
                'text_length': len(text),
                'prediction': pred
            })
        
        self.log_step('predict_batch', 'success', {
            'count': len(predictions)
        })
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    # ============ STEP 3: SAVE PREDICTIONS ============
    
    def save_predictions(self, predictions: List[Dict], batch: bool = True) -> Dict:
        """Save predictions to organized text files"""
        print(f"\n{'='*100}")
        print(f"STEP 3: SAVE PREDICTIONS")
        print(f"{'='*100}")
        
        saved_files = []
        
        # Save individual predictions
        for i, pred_data in enumerate(predictions, 1):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pipeline_prediction_{i:02d}_{timestamp}.txt"
            output_file = os.path.join(self.config['output_dir'], filename)
            
            success = self.predictor.save_prediction_to_file(
                summary=pred_data['prediction'].get('predicted_medicine', 'N/A'),
                date=pred_data['date'],
                output_file=output_file
            )
            
            if success:
                saved_files.append(output_file)
                print(f"  ✓ Saved: {filename}")
        
        # Save batch if requested
        batch_file = None
        if batch and len(predictions) > 1:
            # Extract predictions properly
            texts = [p['prediction'].get('predicted_medicine', 'N/A') for p in predictions]
            dates = [p['date'] for p in predictions]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_file = os.path.join(self.config['output_dir'], f"pipeline_batch_{timestamp}.txt")
            
            self.predictor.save_batch_predictions_to_file(
                summaries=texts,
                dates=dates,
                output_file=batch_file
            )
            saved_files.append(batch_file)
            print(f"  ✓ Saved batch: {Path(batch_file).name}")
        
        self.log_step('save_predictions', 'success', {
            'individual_files': len(predictions),
            'batch_file': batch_file is not None,
            'total_files': len(saved_files)
        })
        
        return {
            'individual_files': [f for f in saved_files if 'batch' not in f],
            'batch_file': batch_file,
            'total_files': len(saved_files)
        }
    
    # ============ STEP 4: TRAIN MODEL ============
    
    def train_model(self, data_source: str = 'training_dataset', model_type: str = 'naive_bayes') -> Dict:
        """Train or update ML model"""
        print(f"\n{'='*100}")
        print(f"STEP 4: TRAIN MODEL")
        print(f"{'='*100}")
        print(f"🤖 Training model ({model_type}) with data from {data_source}...")
        
        try:
            # Load training data
            if data_source == 'training_dataset':
                df = pd.read_csv(os.path.join(self.config['data_dir'], 'training_dataset.csv'))
                print(f"  ✓ Loaded training dataset: {len(df)} records")
            
            elif data_source == 'extracted':
                # Process extracted files
                print(f"  ✓ Processing extracted files...")
                extracted_texts = []
                for file in Path(self.config['output_dir']).glob('*_extracted_*.txt'):
                    with open(file, 'r') as f:
                        extracted_texts.append(f.read())
                
                if not extracted_texts:
                    self.log_error("No extracted files found for training")
                    return {'success': False, 'error': 'No extracted files'}
                
                print(f"  ✓ Found {len(extracted_texts)} extracted files")
                df = pd.DataFrame({'summary': extracted_texts})
            
            else:
                self.log_error(f"Unknown data source: {data_source}")
                return {'success': False, 'error': f'Unknown data source: {data_source}'}
            
            if len(df) == 0:
                self.log_error("No training data available")
                return {'success': False, 'error': 'No training data'}
            
            # Build and train model
            self.model.build_model(model_type=model_type)
            self.model.train(df, model_type=model_type)
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"pipeline_model_{timestamp}"
            self.model.save_model(model_name)
            
            # Update predictor
            self.predictor.load_model(model_name)
            
            self.log_step('train_model', 'success', {
                'data_source': data_source,
                'model_type': model_type,
                'training_samples': len(df),
                'model_name': model_name
            })
            
            print(f"  ✓ Model trained on {len(df)} samples")
            print(f"  ✓ Model saved: {model_name}")
            
            return {
                'success': True,
                'model_name': model_name,
                'training_samples': len(df),
                'model_type': model_type
            }
        
        except Exception as e:
            self.log_error(f"Model training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # ============ COMPLETE PIPELINE ============
    
    def run_complete_pipeline(self, input_file: str, auto_train: bool = None) -> Dict:
        """
        Run complete pipeline for a single file
        
        Steps:
        1. Extract text from document
        2. Generate prediction
        3. Save prediction to file
        4. Optionally train model
        
        Args:
            input_file: Path to input PDF or image
            auto_train: Whether to train model (uses config if None)
        
        Returns:
            Pipeline result dictionary
        """
        if auto_train is None:
            auto_train = self.config['auto_train']
        
        print(f"\n{'='*100}")
        print(f"MEDICAL DATA PIPELINE - COMPLETE WORKFLOW")
        print(f"{'='*100}")
        
        # Step 1: Extract
        extraction = self.extract_from_file(input_file)
        if not extraction['success']:
            return {'success': False, 'error': extraction.get('error')}
        
        extracted_text = extraction['text']
        
        # Step 2: Predict
        print(f"\n{'='*100}")
        print(f"STEP 2: GENERATE PREDICTION")
        print(f"{'='*100}")
        prediction = self.predict_from_text(extracted_text)
        
        # Step 3: Save
        print(f"\n{'='*100}")
        print(f"STEP 3: SAVE RESULTS")
        print(f"{'='*100}")
        
        date = datetime.now().strftime('%Y-%m-%d')
        output_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_file = os.path.join(self.config['output_dir'], output_filename)
        
        self.predictor.save_prediction_to_file(
            summary=extracted_text,
            date=date,
            output_file=output_file
        )
        
        self.log_step('save_results', 'success', {
            'output_file': output_file
        })
        
        print(f"✓ Results saved to: {output_file}")
        
        # Step 4: Optional training
        training_result = None
        if auto_train:
            print(f"\n{'='*100}")
            print(f"STEP 4: OPTIONAL MODEL TRAINING")
            print(f"{'='*100}")
            training_result = self.train_model(
                data_source='training_dataset',
                model_type=self.config['training_model_type']
            )
        
        # Finalize log
        self.log['completed'] = datetime.now().isoformat()
        
        result = {
            'success': True,
            'extraction': {
                'file': input_file,
                'file_type': extraction['file_type'],
                'text_length': len(extracted_text)
            },
            'prediction': prediction,
            'output_file': output_file,
            'training': training_result,
            'log': self.log
        }
        
        print(f"\n{'='*100}")
        print(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*100}\n")
        
        return result
    
    def run_batch_pipeline(self, input_directory: str, auto_train: bool = None) -> Dict:
        """
        Run complete pipeline for all files in a directory
        
        Steps:
        1. Extract all files
        2. Generate predictions for all
        3. Save predictions
        4. Optionally train model
        
        Args:
            input_directory: Directory containing PDF/image files
            auto_train: Whether to train model (uses config if None)
        
        Returns:
            Pipeline result dictionary
        """
        if auto_train is None:
            auto_train = self.config['auto_train']
        
        print(f"\n{'='*100}")
        print(f"MEDICAL DATA PIPELINE - BATCH PROCESSING")
        print(f"{'='*100}")
        
        # Step 1: Extract all files
        extractions = self.extract_from_directory(input_directory)
        
        successful_extractions = [e for e in extractions if e['success']]
        if not successful_extractions:
            return {'success': False, 'error': 'No files extracted'}
        
        # Step 2: Predict all
        print(f"\n{'='*100}")
        print(f"STEP 2: GENERATE PREDICTIONS")
        print(f"{'='*100}")
        
        texts = [e['text'] for e in successful_extractions]
        predictions = self.predict_batch(texts)
        
        # Step 3: Save
        save_result = self.save_predictions(predictions, batch=True)
        
        # Step 4: Optional training
        training_result = None
        if auto_train:
            training_result = self.train_model(
                data_source='extracted',
                model_type=self.config['training_model_type']
            )
        
        # Finalize log
        self.log['completed'] = datetime.now().isoformat()
        
        result = {
            'success': True,
            'files_processed': len(extractions),
            'files_successful': len(successful_extractions),
            'predictions_generated': len(predictions),
            'predictions_saved': save_result['total_files'],
            'training': training_result,
            'log': self.log
        }
        
        print(f"\n{'='*100}")
        print(f"✓ BATCH PIPELINE COMPLETED")
        print(f"{'='*100}\n")
        
        return result
    
    def save_log(self, filename: str = None) -> str:
        """Save pipeline log to JSON file"""
        if filename is None:
            filename = f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_path = os.path.join(self.config['output_dir'], filename)
        
        with open(log_path, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        print(f"Pipeline log saved to: {log_path}")
        return log_path


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = PipelineOrchestrator()
    
    # Run batch pipeline on samples
    if os.path.exists('samples'):
        result = pipeline.run_batch_pipeline('samples', auto_train=True)
        
        # Save log
        pipeline.save_log()
        
        print("\nPipeline Summary:")
        print(f"  Files processed: {result.get('files_processed')}")
        print(f"  Predictions generated: {result.get('predictions_generated')}")
        print(f"  Files saved: {result.get('predictions_saved')}")
        
        if result.get('training'):
            print(f"  Model trained: {result['training']['model_name']}")
    else:
        print("Samples directory not found. Creating example...")
        print("Place PDF or image files in the 'samples' directory and run this script again.")
