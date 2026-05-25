"""
Training script for medical prediction model
Loads extracted data, processes it, and trains the ML model
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.data_processor import MedicalDataProcessor
from ml.model import MedicalPredictionModel


def main():
    """Main training pipeline"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MEDICAL DATA ML MODEL TRAINING PIPELINE" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    # Step 1: Process extracted data
    print("STEP 1: Processing Extracted Medical Data")
    print("-" * 80)
    
    processor = MedicalDataProcessor(output_dir='output', data_dir='data')
    processed_data = processor.process_directory(recursive=True)
    
    if not processed_data:
        print("✗ No data found to process. Please extract files first using extraction_system.py")
        return
    
    # Save processed data
    processor.save_to_csv()
    processor.save_to_json()
    processor.print_summary()
    
    # Step 2: Create training dataset
    print("\n\nSTEP 2: Creating Training Dataset")
    print("-" * 80)
    
    training_df = processor.create_training_dataset(
        output_file=os.path.join('data', 'training_dataset.csv')
    )
    
    print(f"\nDataset shape: {training_df.shape}")
    print(f"Columns: {list(training_df.columns)}")
    
    # Step 3: Build and train model
    print("\n\nSTEP 3: Building and Training ML Model")
    print("-" * 80)
    
    model = MedicalPredictionModel(model_dir='models')
    training_results = model.train(training_df, model_type='naive_bayes')
    
    print("\nTraining Results:")
    for key, value in training_results.items():
        print(f"  {key}: {value}")
    
    # Step 4: Save model
    print("\n\nSTEP 4: Saving Trained Model")
    print("-" * 80)
    
    model_path = model.save_model(model_name='medical_model_v1')
    
    # Step 5: Test predictions
    print("\n\nSTEP 5: Testing Model Predictions")
    print("-" * 80)
    
    # Test with some samples from the data
    test_samples = training_df['summary'].head(3).tolist()
    
    print("\nTest Predictions:")
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. Input Summary:")
        print(f"   {sample[:100]}...")
        
        prediction = model.predict(sample)
        
        if 'error' not in prediction:
            print(f"   Predicted Medicine: {prediction['predicted_medicine']}")
            if prediction['confidence']:
                print(f"   Confidence: {prediction['confidence']:.4f}")
        else:
            print(f"   Error: {prediction['error']}")
    
    # Print model summary
    model.print_model_summary()
    
    # Step 6: Final summary
    print("\nSTEP 6: Training Complete!")
    print("=" * 80)
    print(f"\n✓ Model saved at: {model_path}")
    print(f"✓ Training data: data/training_dataset.csv")
    print(f"✓ Processed data: data/medical_data.csv")
    print(f"✓ Model info: data/medical_data.json")
    print(f"\nNext steps:")
    print(f"  - Use inference_model.py to make predictions")
    print(f"  - Use load_model() to load this trained model")
    print(f"  - Add more training data for better accuracy")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
