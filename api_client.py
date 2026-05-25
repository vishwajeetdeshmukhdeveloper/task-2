"""
API Client for Medical Prediction Pipeline
Examples of how to use the API programmatically
"""

import requests
import json
from pathlib import Path
from typing import Dict, List


class MedicalAPIClient:
    """Client for interacting with Medical Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_docs(self) -> Dict:
        """Get API documentation"""
        response = self.session.get(f"{self.base_url}/api/docs")
        return response.json()
    
    def get_status(self) -> Dict:
        """Get API and system status"""
        response = self.session.get(f"{self.base_url}/api/status")
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        response = self.session.get(f"{self.base_url}/api/model-info")
        return response.json()
    
    def extract(self, file_path: str) -> Dict:
        """
        Extract text from a document
        
        Args:
            file_path: Path to PDF or image file
        
        Returns:
            Extraction result with extracted text
        """
        if not Path(file_path).exists():
            return {'success': False, 'error': 'File not found'}
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(f"{self.base_url}/api/extract", files=files)
        
        return response.json()
    
    def predict(self, summary: str, date: str = None, save_to_file: bool = False) -> Dict:
        """
        Generate prediction from medical summary
        
        Args:
            summary: Medical summary text
            date: Date in YYYY-MM-DD format (optional)
            save_to_file: Whether to save to file (optional)
        
        Returns:
            Prediction result
        """
        data = {
            'summary': summary,
            'save_to_file': save_to_file
        }
        
        if date:
            data['date'] = date
        
        response = self.session.post(
            f"{self.base_url}/api/predict",
            json=data
        )
        
        return response.json()
    
    def batch_predict(self, summaries: List[str], dates: List[str] = None, save_to_file: bool = False) -> Dict:
        """
        Generate predictions for multiple summaries
        
        Args:
            summaries: List of medical summary texts
            dates: List of dates (optional)
            save_to_file: Whether to save to file (optional)
        
        Returns:
            Batch prediction result
        """
        data = {
            'summaries': summaries,
            'save_to_file': save_to_file
        }
        
        if dates:
            data['dates'] = dates
        
        response = self.session.post(
            f"{self.base_url}/api/batch-predict",
            json=data
        )
        
        return response.json()
    
    def pipeline(self, file_path: str) -> Dict:
        """
        Complete pipeline: extract -> predict -> save
        
        Args:
            file_path: Path to PDF or image file
        
        Returns:
            Complete pipeline result
        """
        if not Path(file_path).exists():
            return {'success': False, 'error': 'File not found'}
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(f"{self.base_url}/api/pipeline", files=files)
        
        return response.json()
    
    def train(self, data_source: str = 'training_dataset', model_type: str = 'naive_bayes') -> Dict:
        """
        Train/update ML model
        
        Args:
            data_source: 'training_dataset' or 'extracted'
            model_type: 'naive_bayes' or 'logistic_regression'
        
        Returns:
            Training result with accuracy
        """
        data = {
            'data_source': data_source,
            'model_type': model_type
        }
        
        response = self.session.post(
            f"{self.base_url}/api/train",
            json=data
        )
        
        return response.json()


# Example usage functions

def example_1_extract():
    """Example 1: Extract text from a document"""
    print("\n" + "=" * 100)
    print("EXAMPLE 1: EXTRACT TEXT FROM DOCUMENT")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    # Replace with your file path
    file_path = "samples/report1.pdf"
    
    print(f"📄 Extracting from: {file_path}")
    
    result = client.extract(file_path)
    
    if result['success']:
        print(f"✓ Extraction successful!")
        print(f"  File type: {result['file_type']}")
        print(f"  Text length: {result['text_length']} characters")
        print(f"  Preview: {result['text'][:200]}...")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_2_predict():
    """Example 2: Generate prediction from text"""
    print("\n" + "=" * 100)
    print("EXAMPLE 2: GENERATE PREDICTION")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    summary = "Patient shows normal blood work. All values within range. Hematocrit 42%, Hemoglobin 13.5 g/dL."
    
    print(f"📊 Generating prediction for summary...")
    
    result = client.predict(summary, save_to_file=True)
    
    if result['success']:
        pred = result['prediction']
        print(f"✓ Prediction generated!")
        print(f"  Recommendation: {pred.get('predicted_medicine')}")
        print(f"  Confidence: {pred.get('confidence', 'N/A')}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_3_batch_predict():
    """Example 3: Batch predictions"""
    print("\n" + "=" * 100)
    print("EXAMPLE 3: BATCH PREDICTIONS")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    summaries = [
        "Normal hemoglobin levels detected. Patient is healthy.",
        "Elevated white blood cell count. Possible infection.",
        "Thyroid function abnormal. TSH levels high."
    ]
    
    print(f"📊 Generating {len(summaries)} predictions...")
    
    result = client.batch_predict(summaries, save_to_file=True)
    
    if result['success']:
        print(f"✓ {result['count']} predictions generated!")
        print(f"✓ Saved to file: {result.get('output_file')}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_4_complete_pipeline():
    """Example 4: Complete pipeline"""
    print("\n" + "=" * 100)
    print("EXAMPLE 4: COMPLETE PIPELINE (Extract -> Predict -> Save)")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    file_path = "samples/report1.pdf"
    
    print(f"🔄 Running complete pipeline for: {file_path}")
    
    result = client.pipeline(file_path)
    
    if result['success']:
        print(f"✓ Pipeline completed!")
        print(f"  Extraction: {result['extraction']['file_type']}")
        print(f"  Text length: {result['extraction']['text_length']}")
        pred = result['prediction']
        print(f"  Prediction: {pred.get('predicted_medicine')}")
        print(f"  Output file: {result['output_file']}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_5_train_model():
    """Example 5: Train model"""
    print("\n" + "=" * 100)
    print("EXAMPLE 5: TRAIN/UPDATE MODEL")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    print(f"🤖 Training model with data source: training_dataset")
    
    result = client.train(data_source='training_dataset', model_type='naive_bayes')
    
    if result['success']:
        print(f"✓ Model training completed!")
        print(f"  Model name: {result['model_name']}")
        print(f"  Training samples: {result['samples_used']}")
        print(f"  Accuracy: {result['accuracy']}")
    else:
        print(f"✗ Error: {result.get('error')}")


def example_6_status_and_info():
    """Example 6: Check status and model info"""
    print("\n" + "=" * 100)
    print("EXAMPLE 6: API STATUS AND MODEL INFORMATION")
    print("=" * 100 + "\n")
    
    client = MedicalAPIClient()
    
    # Health check
    print("🏥 Health Check:")
    health = client.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Timestamp: {health['timestamp']}")
    
    # API Status
    print("\n🔍 API Status:")
    status = client.get_status()
    print(f"  API Status: {status['api_status']}")
    print(f"  Directories:")
    for dir_name, exists in status['directories'].items():
        print(f"    - {dir_name}: {'✓' if exists else '✗'}")
    
    # Model info
    print("\n📊 Model Information:")
    model_info = client.get_model_info()
    print(f"  Model type: {model_info['model_type']}")
    print(f"  Features: {model_info['features']}")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 98 + "╗")
    print("║" + " " * 20 + "MEDICAL PREDICTION API - CLIENT EXAMPLES" + " " * 38 + "║")
    print("╚" + "=" * 98 + "╝")
    
    print("\n⚠️  Make sure API is running: python api.py")
    print("⏳ Waiting for API to be ready...")
    
    import time
    
    client = MedicalAPIClient()
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            health = client.health_check()
            print(f"✓ API is ready!\n")
            break
        except:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1}/{max_retries}... Retrying in 2 seconds")
                time.sleep(2)
            else:
                print("✗ Could not connect to API. Make sure it's running on http://localhost:5000")
                return
    
    # Run examples
    try:
        example_1_extract()
        example_2_predict()
        example_3_batch_predict()
        example_4_complete_pipeline()
        example_5_train_model()
        example_6_status_and_info()
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        print("Make sure all sample files exist and API is running properly")
    
    print("\n" + "=" * 100)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
