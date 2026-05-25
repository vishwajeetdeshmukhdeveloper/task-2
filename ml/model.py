"""
Machine Learning Model for medical data prediction
Trains and predicts: Date - Summary, Medicine Recommendation
"""

import os
import joblib
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MedicalPredictionModel:
    """
    ML model for predicting medicine recommendations based on medical summaries
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize model
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.training_info = {}
    
    def build_model(self, model_type: str = 'naive_bayes') -> None:
        """
        Build ML model pipeline
        
        Args:
            model_type: Type of model ('naive_bayes', 'random_forest')
        """
        print(f"Building {model_type} model...")
        
        # Text vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True,
            stop_words='english'
        )
        
        # Select base model
        if model_type == 'naive_bayes':
            base_model = MultinomialNB(alpha=0.1)
        elif model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = base_model
        self.training_info['model_type'] = model_type
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with 'combined_text' and 'medicine_recommendation' columns
            
        Returns:
            Tuple of (X_vectorized, y_medicine_recommendations)
        """
        # Vectorize text
        X = self.vectorizer.fit_transform(df['combined_text'].fillna(''))
        y = df['medicine_recommendation'].fillna('Standard care').tolist()
        
        return X, y
    
    def train(self, df: pd.DataFrame, model_type: str = 'naive_bayes') -> Dict:
        """
        Train the model
        
        Args:
            df: Training DataFrame
            model_type: Type of model to build
            
        Returns:
            Training results dictionary
        """
        print("\n" + "=" * 80)
        print("TRAINING MEDICAL PREDICTION MODEL")
        print("=" * 80)
        
        if len(df) < 5:
            print(f"Warning: Only {len(df)} records. Model may not generalize well.")
        
        # Build model
        self.build_model(model_type)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Convert medicine recommendations to labels for classification
        unique_medicines = list(set(y))
        medicine_to_idx = {med: idx for idx, med in enumerate(unique_medicines)}
        y_encoded = np.array([medicine_to_idx[med] for med in y])
        
        print(f"Training samples: {len(y)}")
        print(f"Features: {X.shape[1]}")
        print(f"Medicine categories: {len(unique_medicines)}")
        
        # Train model
        self.model.fit(X, y_encoded)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_encoded, y_pred)
        
        self.is_trained = True
        self.training_info['timestamp'] = datetime.now().isoformat()
        self.training_info['train_samples'] = len(y)
        self.training_info['features'] = X.shape[1]
        self.training_info['accuracy'] = float(accuracy)
        self.training_info['classes'] = unique_medicines
        
        results = {
            'success': True,
            'accuracy': accuracy,
            'samples': len(y),
            'features': X.shape[1],
            'classes': len(unique_medicines),
            'model_type': model_type
        }
        
        print(f"\n✓ Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Medicine categories: {len(unique_medicines)}")
        
        return results
    
    def predict(self, text: str) -> Dict:
        """
        Predict medicine recommendation for given medical summary
        
        Args:
            text: Medical summary text
            
        Returns:
            Dictionary with predictions
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Predict
            pred_idx = self.model.predict(X)[0]
            
            # Get predicted class
            predicted_medicine = self.training_info['classes'][pred_idx]
            
            # Get prediction confidence if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                confidence = float(max(probs))
            
            return {
                'input_text': text,
                'predicted_medicine': predicted_medicine,
                'confidence': confidence,
                'prediction_time': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'input_text': text
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict for multiple texts
        
        Args:
            texts: List of medical summary texts
            
        Returns:
            List of predictions
        """
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            model_name: Optional model name
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if model_name is None:
            model_name = f"medical_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        vectorizer_path = os.path.join(self.model_dir, f"{model_name}_vectorizer.pkl")
        info_path = os.path.join(self.model_dir, f"{model_name}_info.pkl")
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save vectorizer
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save info
            with open(info_path, 'wb') as f:
                pickle.dump(self.training_info, f)
            
            print(f"✓ Model saved: {model_path}")
            print(f"✓ Vectorizer saved: {vectorizer_path}")
            print(f"✓ Info saved: {info_path}")
            
            return model_path
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return ""
    
    def load_model(self, model_name: str) -> bool:
        """
        Load trained model
        
        Args:
            model_name: Model name (without extension)
            
        Returns:
            True if successful
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        vectorizer_path = os.path.join(self.model_dir, f"{model_name}_vectorizer.pkl")
        info_path = os.path.join(self.model_dir, f"{model_name}_info.pkl")
        
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, info_path]):
            print(f"Model files not found for: {model_name}")
            return False
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load info
            with open(info_path, 'rb') as f:
                self.training_info = pickle.load(f)
            
            self.is_trained = True
            
            print(f"✓ Model loaded: {model_name}")
            print(f"  Type: {self.training_info.get('model_type')}")
            print(f"  Accuracy: {self.training_info.get('accuracy'):.4f}")
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return self.training_info
    
    def print_model_summary(self) -> None:
        """Print model summary"""
        if not self.is_trained:
            print("Model not trained yet")
            return
        
        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        print(f"Model Type: {self.training_info.get('model_type')}")
        print(f"Trained: {self.training_info.get('timestamp')}")
        print(f"Training Samples: {self.training_info.get('train_samples')}")
        print(f"Features: {self.training_info.get('features')}")
        print(f"Accuracy: {self.training_info.get('accuracy'):.4f}")
        print(f"Classes: {len(self.training_info.get('classes', []))}")
        print("\nMedicine Categories:")
        for i, med in enumerate(self.training_info.get('classes', [])[:5], 1):
            print(f"  {i}. {med[:60]}...")
        if len(self.training_info.get('classes', [])) > 5:
            print(f"  ... and {len(self.training_info.get('classes', [])) - 5} more")
        print("=" * 80 + "\n")
