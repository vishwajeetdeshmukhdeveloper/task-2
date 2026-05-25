"""
Inference script - Use trained model to make predictions
Format: Date - Summary, Medicine Recommendation
"""

import os
import sys
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.model import MedicalPredictionModel
from ml.data_processor import MedicalDataProcessor
from medical_text_cleaner import MedicalTextCleaner


class MedicalPredictor:
    """
    Predictor class that loads model and makes predictions
    Output format: YYYY-MM-DD - Summary, Medicine Recommendation
    """
    
    def __init__(self, model_name: str = 'medical_model_v1', model_dir: str = 'models'):
        """
        Initialize predictor
        
        Args:
            model_name: Name of the trained model
            model_dir: Directory containing models
        """
        self.model = MedicalPredictionModel(model_dir=model_dir)
        self.model_loaded = self.model.load_model(model_name)
        
        if not self.model_loaded:
            print("Warning: Model not found. Train a model first using train_model.py")
    
    def predict_single(self, summary: str, date: str = None) -> str:
        """
        Make a single prediction
        
        Args:
            summary: Medical summary text
            date: Optional date (format: YYYY-MM-DD)
            
        Returns:
            Formatted prediction: Date - Summary, Medicine Recommendation
        """
        if not self.model_loaded:
            return "Error: Model not loaded"
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Get prediction
        prediction = self.model.predict(summary)
        
        if 'error' in prediction:
            return f"Error: {prediction['error']}"
        
        # Format output
        medicine = prediction['predicted_medicine']
        output = f"{date} - {summary}, {medicine}"
        
        return output
    
    def predict_batch(self, summaries: List[str], dates: List[str] = None) -> List[str]:
        """
        Make batch predictions
        
        Args:
            summaries: List of medical summaries
            dates: Optional list of dates
            
        Returns:
            List of formatted predictions
        """
        if not self.model_loaded:
            return ["Error: Model not loaded"]
        
        if dates is None:
            dates = [datetime.now().strftime('%Y-%m-%d')] * len(summaries)
        
        predictions = []
        for summary, date in zip(summaries, dates):
            pred = self.predict_single(summary, date)
            predictions.append(pred)
        
        return predictions
    
    def predict_from_csv(self, csv_file: str, summary_column: str = 'summary', 
                        date_column: str = 'date', output_file: str = None) -> bool:
        """
        Make predictions from CSV file
        
        Args:
            csv_file: Path to CSV file with summaries
            summary_column: Column name containing summaries
            date_column: Column name containing dates
            output_file: Optional output file path
            
        Returns:
            True if successful
        """
        if not self.model_loaded:
            print("Error: Model not loaded")
            return False
        
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if summary_column not in df.columns:
                print(f"Error: Column '{summary_column}' not found in CSV")
                return False
            
            # Make predictions
            predictions = []
            for idx, row in df.iterrows():
                summary = row[summary_column]
                date = row[date_column] if date_column in df.columns else datetime.now().strftime('%Y-%m-%d')
                
                pred = self.predict_single(summary, str(date))
                predictions.append(pred)
            
            # Save predictions
            if output_file:
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("Predictions (Date - Summary, Medicine Recommendation)\n")
                    f.write("=" * 100 + "\n\n")
                    for pred in predictions:
                        f.write(pred + "\n\n")
                
                print(f"✓ Predictions saved to: {output_file}")
            
            return True
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return False
    
    def print_prediction(self, date: str, summary: str, medicine: str) -> None:
        """Print formatted prediction"""
        print("\n" + "=" * 100)
        print("PREDICTION RESULT")
        print("=" * 100)
        print(f"\nDate: {date}")
        print(f"Summary: {summary}")
        print(f"Medicine Recommendation: {medicine}")
        print("\n" + "=" * 100)
    
    def save_prediction_to_file(self, summary: str, date: str = None, output_file: str = None, recommendation: str = None) -> bool:
        """
        Save a single prediction to organized text file
        Format: 
            Date: YYYY-MM-DD
            Summary: Medical summary text
            Recommendation: Medicine recommendation
        
        Args:
            summary: Medical summary text (will be cleaned if not pre-cleaned)
            date: Optional date (format: YYYY-MM-DD)
            output_file: Output file path
            recommendation: Optional pre-computed recommendation
            
        Returns:
            True if successful
        """
        if not self.model_loaded:
            print("Error: Model not loaded")
            return False
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if output_file is None:
            output_file = os.path.join('output', f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        try:
            # If already clean (short) and has recommendation, use as-is
            # Otherwise, clean it
            if len(summary) > 300 or not recommendation:
                cleaner = MedicalTextCleaner()
                clean_summary, recommended_action = cleaner.clean_summary(summary)
            else:
                clean_summary = summary
                recommended_action = recommendation
            
            # Get prediction
            prediction = self.model.predict(clean_summary)
            
            if 'error' in prediction:
                print(f"Error: {prediction['error']}")
                return False
            
            medicine = prediction.get('predicted_medicine', recommended_action)
            if recommendation:
                medicine = recommendation
            confidence = prediction.get('confidence', 0)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            # Write organized format
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("MEDICAL PREDICTION REPORT\n")
                f.write("=" * 100 + "\n\n")
                
                f.write(f"Date: {date}\n")
                f.write(f"\nSummary:\n{clean_summary}\n")
                f.write(f"\nRecommendation:\n{medicine}\n\n")
                
                f.write("=" * 100 + "\n")
            
            print(f"✓ Prediction saved to: {output_file}")
            return True
        
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            return False
    
    def save_batch_predictions_to_file(self, summaries: List[str], dates: List[str] = None, 
                                       output_file: str = None) -> bool:
        """
        Save multiple predictions to organized text file
        
        Args:
            summaries: List of medical summaries
            dates: Optional list of dates
            output_file: Output file path
            
        Returns:
            True if successful
        """
        if not self.model_loaded:
            print("Error: Model not loaded")
            return False
        
        if dates is None:
            dates = [datetime.now().strftime('%Y-%m-%d')] * len(summaries)
        
        if output_file is None:
            output_file = os.path.join('output', f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            # Initialize text cleaner for cleaning summaries
            cleaner = MedicalTextCleaner()
            
            # Write all predictions
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write(f"BATCH MEDICAL PREDICTIONS REPORT - {len(summaries)} Records\n")
                f.write("=" * 100 + "\n\n")
                
                for i, (summary, date) in enumerate(zip(summaries, dates), 1):
                    # Clean the summary using medical text cleaner
                    cleaned_summary, recommendation = cleaner.clean_summary(summary)
                    
                    prediction = self.model.predict(summary)
                    
                    f.write(f"Prediction #{i}\n")
                    f.write("-" * 100 + "\n")
                    
                    if 'error' not in prediction:
                        medicine = prediction['predicted_medicine']
                        f.write(f"Date: {date}\n")
                        f.write(f"\nSummary:\n{cleaned_summary}\n")
                        f.write(f"\nRecommendation:\n{medicine}\n")
                    else:
                        f.write(f"Error: {prediction['error']}\n")
                    
                    f.write("\n")
                
                f.write("=" * 100 + "\n")
                f.write(f"Total Predictions: {len(summaries)}\n")
                f.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n")
            
            print(f"✓ Batch predictions saved to: {output_file}")
            print(f"  Total records: {len(summaries)}")
            return True
        
        except Exception as e:
            print(f"Error saving batch predictions: {str(e)}")
            return False


def example_interactive():
    """Interactive prediction example"""
    
    print("\n" + "╔" + "=" * 98 + "╗")
    print("║" + " " * 30 + "MEDICAL PREDICTION SYSTEM" + " " * 44 + "║")
    print("║" + " " * 15 + "Format: Date - Summary, Medicine Recommendation" + " " * 36 + "║")
    print("╚" + "=" * 98 + "╝\n")
    
    # Initialize predictor
    predictor = MedicalPredictor()
    
    if not predictor.model_loaded:
        print("Please train a model first using: python train_model.py\n")
        return
    
    # Example predictions
    print("EXAMPLE PREDICTIONS:")
    print("-" * 100)
    
    examples = [
        {
            'date': '2026-05-15',
            'summary': 'Complete blood count normal. All values within range. Patient shows good health status.',
            'medicine': None
        },
        {
            'date': '2026-05-14',
            'summary': 'Elevated white blood cell count detected. Possible infection. Abnormal findings require follow-up.',
            'medicine': None
        },
        {
            'date': '2026-05-13',
            'summary': 'Blood glucose level elevated. Diabetes risk detected. Immediate dietary intervention recommended.',
            'medicine': None
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Input Summary:")
        print(f"   {example['summary']}")
        
        prediction = predictor.model.predict(example['summary'])
        
        if 'error' not in prediction:
            medicine = prediction['predicted_medicine']
            output = f"{example['date']} - {example['summary']}, {medicine}"
            print(f"\n   Output Format:")
            print(f"   {output}")
            
            if prediction['confidence']:
                print(f"   Confidence: {prediction['confidence']:.2%}")
        else:
            print(f"   Error: {prediction['error']}")
    
    print("\n" + "=" * 100 + "\n")


def main():
    """Main function"""
    
    print("\n" + "═" * 100)
    print("MEDICAL DATA PREDICTION MODEL - INFERENCE")
    print("═" * 100 + "\n")
    
    # Option 1: Interactive demo
    print("Running Example Predictions...")
    example_interactive()
    
    # Option 2: Predict from extracted training data
    print("\nPredicting from Training Data...")
    print("-" * 100)
    
    predictor = MedicalPredictor()
    
    if predictor.model_loaded:
        # Load training data
        try:
            import pandas as pd
            
            training_file = 'data/training_dataset.csv'
            if os.path.exists(training_file):
                df = pd.read_csv(training_file)
                
                # Make predictions on first few samples
                print(f"Making predictions on {min(3, len(df))} training samples:\n")
                
                for idx in range(min(3, len(df))):
                    summary = df.iloc[idx]['summary']
                    date = df.iloc[idx]['date'] if 'date' in df.columns else datetime.now().strftime('%Y-%m-%d')
                    
                    output = predictor.predict_single(summary, str(date))
                    print(f"{idx+1}. {output}\n")
            
        except Exception as e:
            print(f"Note: {str(e)}")
    
    print("\n" + "═" * 100)
    print("To make custom predictions, use:")
    print("  predictor = MedicalPredictor()")
    print("  result = predictor.predict_single('Your medical summary here')")
    print("═" * 100 + "\n")


if __name__ == "__main__":
    main()
