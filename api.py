"""
Medical Prediction API
Flask API for document extraction, prediction, and model training
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from pathlib import Path
import json

# Import custom modules
from text_extraction import TextExtractor
from inference_model import MedicalPredictor
from medical_text_cleaner import MedicalTextCleaner
from ml.data_processor import MedicalDataProcessor
from ml.model import MedicalPredictionModel
import pandas as pd

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'jpg', 'jpeg', 'png', 'bmp'}

# Initialize components
extractor = TextExtractor(output_dir='output')
predictor = MedicalPredictor()

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def reload_predictor():
    """Reload predictor to ensure fresh model"""
    global predictor
    predictor = MedicalPredictor()
    return predictor


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Medical Prediction API'
    }), 200


@app.route('/api/extract', methods=['POST'])
def extract_text():
    """
    Extract text from uploaded document (PDF or Image)
    
    Request:
        - file: Document file (PDF, JPG, PNG, BMP)
    
    Response:
        - success: Boolean
        - file_type: Type of file processed
        - text: Extracted text
        - file_path: Path to uploaded file
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text
        result = extractor.extract_file(file_path)
        
        return jsonify({
            'success': result['success'],
            'file_type': result.get('file_type'),
            'text': result.get('text', '')[:1000] + '...' if result.get('text') and len(result.get('text', '')) > 1000 else result.get('text', ''),
            'text_length': len(result.get('text', '')),
            'file_path': file_path,
            'error': result.get('error'),
            'extracted_at': result.get('extracted_at')
        }), 200 if result['success'] else 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate prediction from medical summary
    
    Request JSON:
        - summary: Medical summary text (required)
        - date: Date of medical record (optional, YYYY-MM-DD format)
        - save_to_file: Save prediction to file (optional, boolean)
    
    Response:
        - success: Boolean
        - predicted_medicine: Recommendation
        - confidence: Confidence percentage
        - date: Date of prediction
        - timestamp: When prediction was made
    """
    try:
        data = request.get_json()
        
        if not data or 'summary' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: summary'
            }), 400
        
        summary = data['summary']
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        save_to_file = data.get('save_to_file', False)
        
        # Clean the input text
        cleaner = MedicalTextCleaner()
        clean_summary, recommended_action = cleaner.clean_summary(summary)
        
        # Generate prediction using model
        prediction = predictor.model.predict(clean_summary)
        
        if 'error' in prediction:
            return jsonify({
                'success': False,
                'error': prediction['error']
            }), 400
        
        # Save to file if requested
        output_file = None
        if save_to_file:
            filename = f"api_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            output_file = os.path.join('output', filename)
            predictor.save_prediction_to_file(
                summary=summary,
                date=date,
                output_file=output_file
            )
        
        return jsonify({
            'success': True,
            'predicted_medicine': prediction.get('predicted_medicine', recommended_action),
            'confidence': prediction.get('confidence', 0),
            'date': date,
            'summary': clean_summary,
            'saved_to_file': save_to_file,
            'output_file': output_file,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/pipeline', methods=['POST'])
def pipeline():
    """
    Complete pipeline: Extract -> Predict -> Save
    
    Request:
        - file: Document file (PDF or Image)
    
    Response:
        - success: Boolean
        - extraction: Extraction results
        - prediction: Prediction results
        - output_file: Saved prediction file path
    """
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file'
            }), 400
        
        # 1. EXTRACT
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        extraction = extractor.extract_file(file_path)
        
        if not extraction['success']:
            return jsonify({
                'success': False,
                'error': extraction.get('error', 'Extraction failed')
            }), 400
        
        extracted_text = extraction['text']
        
        # 2. CLEAN & PREDICT
        current_date = datetime.now().strftime('%Y-%m-%d')
        cleaner = MedicalTextCleaner()
        clean_summary, recommended_action = cleaner.clean_summary(extracted_text)
        
        # Get prediction on cleaned text
        prediction = predictor.model.predict(clean_summary)
        
        # 3. SAVE
        output_filename = f"pipeline_{timestamp}.txt"
        output_file = os.path.join('output', output_filename)
        predictor.save_prediction_to_file(
            summary=clean_summary,  # Use cleaned summary, not raw extracted text
            date=current_date,
            output_file=output_file,
            recommendation=recommended_action
        )
        
        return jsonify({
            'success': True,
            'extraction': {
                'file_type': extraction.get('file_type'),
                'text_length': len(extracted_text),
                'pages': extraction.get('page_count'),
                'text': clean_summary  # Use cleaned summary in preview
            },
            'prediction': {
                'predicted_medicine': prediction.get('predicted_medicine', recommended_action),
                'confidence': prediction.get('confidence', 0),
                'date': current_date,
                'summary': clean_summary
            },
            'output_file': output_file,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train/Update the ML model with new data
    
    Request JSON (optional):
        - data_source: 'training_dataset' or 'extracted' (default: 'training_dataset')
        - model_type: 'naive_bayes' or 'logistic_regression' (default: 'naive_bayes')
    
    Response:
        - success: Boolean
        - samples_used: Number of training samples
        - accuracy: Model accuracy
        - model_saved: Path to saved model
    """
    try:
        data = request.get_json() or {}
        data_source = data.get('data_source', 'training_dataset')
        model_type = data.get('model_type', 'naive_bayes')
        
        # Load training data
        if data_source == 'training_dataset':
            df = pd.read_csv('data/training_dataset.csv')
        elif data_source == 'extracted':
            # Process extracted files
            processor = MedicalDataProcessor()
            df = processor.process_directory('output')
            df = processor.create_training_dataset(df)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown data source: {data_source}'
            }), 400
        
        if len(df) == 0:
            return jsonify({
                'success': False,
                'error': 'No training data available'
            }), 400
        
        # Train model
        model = MedicalPredictionModel()
        model.build_model(model_type=model_type)
        model.train(df, model_type=model_type)
        
        # Save model
        model_name = f"medical_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model.save_model(model_name)
        
        # Update predictor with new model
        predictor.load_model(model_name)
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        predictions = [model.predict(row['summary']) for _, row in df.iterrows()]
        accuracy = accuracy_score(
            [row['medicine_recommendation'] for _, row in df.iterrows()],
            predictions
        )
        
        return jsonify({
            'success': True,
            'samples_used': len(df),
            'accuracy': f"{accuracy * 100:.2f}%",
            'model_type': model_type,
            'model_name': model_name,
            'model_saved': f"models/{model_name}.pkl",
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple summaries
    
    Request JSON:
        - summaries: List of medical summaries
        - dates: List of dates (optional)
        - save_to_file: Save batch predictions (optional, boolean)
    
    Response:
        - success: Boolean
        - predictions: List of predictions
        - output_file: Batch file path (if saved)
    """
    try:
        data = request.get_json()
        
        if not data or 'summaries' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: summaries'
            }), 400
        
        summaries = data['summaries']
        dates = data.get('dates', [datetime.now().strftime('%Y-%m-%d')] * len(summaries))
        save_to_file = data.get('save_to_file', False)
        
        if not isinstance(summaries, list):
            return jsonify({
                'success': False,
                'error': 'summaries must be a list'
            }), 400
        
        # Generate predictions
        predictions = []
        for summary, date in zip(summaries, dates):
            pred = predictor.model.predict(summary)
            if 'error' not in pred:
                predictions.append({
                    'summary': summary[:200],
                    'predicted_medicine': pred.get('predicted_medicine', 'Unknown'),
                    'confidence': pred.get('confidence', 0),
                    'date': date
                })
        
        # Save to file if requested
        output_file = None
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_predictions_{timestamp}.txt"
            output_file = os.path.join('output', filename)
            predictor.save_batch_predictions_to_file(
                summaries=summaries,
                dates=dates,
                output_file=output_file
            )
        
        return jsonify({
            'success': True,
            'count': len(predictions),
            'predictions': predictions,
            'output_file': output_file,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get current model information"""
    try:
        model = predictor.model
        
        return jsonify({
            'success': True,
            'model_type': 'Naive Bayes with TF-IDF',
            'features': model.vectorizer.get_feature_names_out().shape[0] if hasattr(model.vectorizer, 'get_feature_names_out') else 'N/A',
            'model_location': 'models/medical_model_v1.pkl',
            'vectorizer_location': 'models/medical_model_v1_vectorizer.pkl',
            'last_updated': 'Check model file metadata',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get API and system status"""
    try:
        # Check if key directories exist
        status_info = {
            'api_status': 'running',
            'timestamp': datetime.now().isoformat(),
            'directories': {
                'uploads': os.path.exists(app.config['UPLOAD_FOLDER']),
                'output': os.path.exists('output'),
                'models': os.path.exists('models'),
                'data': os.path.exists('data')
            },
            'endpoints': {
                'health': '/health',
                'extract': '/api/extract',
                'predict': '/api/predict',
                'batch_predict': '/api/batch-predict',
                'pipeline': '/api/pipeline',
                'train': '/api/train',
                'model_info': '/api/model-info',
                'status': '/api/status'
            }
        }
        
        return jsonify(status_info), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/docs', methods=['GET'])
def docs():
    """Get API documentation"""
    docs = {
        'title': 'Medical Prediction API',
        'version': '1.0.0',
        'description': 'Extract medical data from documents and generate predictions with ML model',
        'endpoints': {
            'GET /health': {
                'description': 'Health check',
                'returns': 'Service status'
            },
            'POST /api/extract': {
                'description': 'Extract text from PDF or image',
                'params': {'file': 'Document file (multipart/form-data)'},
                'returns': 'Extracted text and metadata'
            },
            'POST /api/predict': {
                'description': 'Generate prediction from medical summary',
                'params': {
                    'summary': 'Medical text (required)',
                    'date': 'Date YYYY-MM-DD (optional)',
                    'save_to_file': 'Boolean (optional)'
                },
                'returns': 'Prediction with confidence'
            },
            'POST /api/batch-predict': {
                'description': 'Generate predictions for multiple summaries',
                'params': {
                    'summaries': 'List of text (required)',
                    'dates': 'List of dates (optional)',
                    'save_to_file': 'Boolean (optional)'
                },
                'returns': 'List of predictions'
            },
            'POST /api/pipeline': {
                'description': 'Complete pipeline: extract -> predict -> save',
                'params': {'file': 'Document file (multipart/form-data)'},
                'returns': 'Extraction and prediction results'
            },
            'POST /api/train': {
                'description': 'Train/update ML model',
                'params': {
                    'data_source': 'training_dataset or extracted (optional)',
                    'model_type': 'naive_bayes or logistic_regression (optional)'
                },
                'returns': 'Model training results and accuracy'
            },
            'GET /api/model-info': {
                'description': 'Get current model information',
                'returns': 'Model metadata'
            },
            'GET /api/status': {
                'description': 'Get API and system status',
                'returns': 'System status and configuration'
            },
            'GET /api/docs': {
                'description': 'Get this documentation',
                'returns': 'API documentation'
            }
        }
    }
    
    return jsonify(docs), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': '/api/docs'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 100)
    print("Medical Prediction API")
    print("=" * 100)
    print("\nStarting API server...")
    print("📍 API running on: http://localhost:5000")
    print("📚 API docs available at: http://localhost:5000/api/docs")
    print("\nAvailable endpoints:")
    print("  ✓ GET  /health - Health check")
    print("  ✓ POST /api/extract - Extract text from document")
    print("  ✓ POST /api/predict - Generate prediction")
    print("  ✓ POST /api/batch-predict - Batch predictions")
    print("  ✓ POST /api/pipeline - Complete pipeline")
    print("  ✓ POST /api/train - Train model")
    print("  ✓ GET  /api/model-info - Model information")
    print("  ✓ GET  /api/status - API status")
    print("\n" + "=" * 100 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
