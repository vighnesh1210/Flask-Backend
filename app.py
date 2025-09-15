from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from PIL import Image

# Import utility modules
from utils.image_utils import ImageProcessor
from utils.model_predictor import ModelPredictor
from utils.ocr_validator import OCRValidator
from utils.gradcam import generate_ocr_focused_gradcam, generate_gradcam, generate_error_focused_gradcam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize components
image_processor = ImageProcessor()
model_predictor = ModelPredictor(model_path=os.path.join(MODEL_FOLDER, 'fake_note_detector_final.h5'))
ocr_validator = OCRValidator()

# Add model summary and debugging info
if model_predictor.is_model_loaded():
    print("\n" + "="*50)
    print("MODEL SUMMARY:")
    print("="*50)
    model_predictor.model.summary()
    print(f"\nModel output shape: {model_predictor.model.output_shape}")
    print(f"Model input shape: {model_predictor.model.input_shape}")
    print("\nLayer names for GradCAM:")
    for i, layer in enumerate(model_predictor.model.layers):
        print(f"{i}: {layer.name} - {layer.__class__.__name__}")
        if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
            print(f"   ^^ This is a Conv layer - shape: {layer.output_shape}")
    print("="*50)
else:
    print("Model failed to load - cannot show summary")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'model_loaded': model_predictor.is_model_loaded(),
            'image_processor': True,
            'ocr_validator': True,
            'gradcam_available': True
        }
    })

@app.route('/authenticate', methods=['POST'])
def authenticate_banknote():
    """Main endpoint for banknote authentication with conditional OCR/GradCAM"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process image
        processed_image = image_processor.preprocess_for_model(file_path)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Get model prediction FIRST
        prediction_result = model_predictor.predict(processed_image)
        
        # Check if note is confidently REAL - if so, skip OCR and GradCAM
        is_real = prediction_result['is_real']
        confidence = prediction_result['confidence']
        
        if is_real and confidence >= model_predictor.confidence_threshold_real:
            # REAL note with high confidence - NO OCR, NO GradCAM
            result = "REAL"
            authenticity_score = confidence
            explanation = f"Authentic banknote detected with high confidence (>{model_predictor.confidence_threshold_real*100}%). No further analysis needed."
            
            response = {
                'result': result,
                'confidence': round(confidence * 100, 2),
                'authenticity_score': round(authenticity_score * 100, 2),
                'explanation': explanation,
                'ocr_analysis': {
                    'status': 'skipped',
                    'reason': 'High confidence authentic note - OCR analysis not required'
                },
                'gradcam_available': False,
                'gradcam_filename': None,
                'gradcam_type': 'none',
                'timestamp': datetime.now().isoformat(),
                'processing_details': {
                    'raw_prediction': prediction_result['raw_prediction'],
                    'model_confidence': prediction_result['confidence'],
                    'ocr_skipped': True,
                    'gradcam_skipped': True,
                    'processing_mode': 'cnn_only'
                }
            }
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            print(f"✅ High confidence REAL note detected ({confidence*100:.1f}%) - Skipping OCR and GradCAM")
            return jsonify(response)
        
        # If we reach here, note is either FAKE, UNCERTAIN, or REAL with low confidence
        # Proceed with OCR and GradCAM analysis
        print(f"🔍 Running full analysis - Prediction: {'REAL' if is_real else 'FAKE'} with {confidence*100:.1f}% confidence")
        
        image_pil = Image.open(file_path)
        comprehensive_ocr_results = ocr_validator.validate_comprehensive_text(image_pil)
        clean_ocr_results = ocr_validator.validate_banknote(file_path)
        
        gradcam_filename = None
        
        if not is_real and confidence >= model_predictor.confidence_threshold_fake:
            # FAKE note
            result = "FAKE"
            authenticity_score = 1 - confidence
            
            print("Generating OCR-focused GradCAM for FAKE note...")
            print(f"OCR results for FAKE note: {len(comprehensive_ocr_results.get('missing_texts', []))} missing texts, {len(comprehensive_ocr_results.get('forbidden_texts_found', []))} forbidden texts")
            
            heatmap = generate_ocr_focused_gradcam(model_predictor.model, image_pil, comprehensive_ocr_results)
            gradcam_filename = f"{uuid.uuid4().hex}_ocr_focused_gradcam.png"
            gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
            Image.fromarray(heatmap).save(gradcam_path)
            explanation = "Counterfeit banknote detected. OCR analysis and heatmap show specific problem areas."
            
        elif is_real and confidence < model_predictor.confidence_threshold_real:
            # REAL but with low confidence - run OCR to check for issues
            result = "REAL"
            authenticity_score = confidence
            
            print(f"Analyzing low-confidence REAL note ({confidence*100:.1f}%) with OCR...")
            
            # Check if OCR found any issues
            if comprehensive_ocr_results.get('missing_texts') or comprehensive_ocr_results.get('forbidden_texts_found'):
                print("Generating OCR-focused GradCAM for low-confidence REAL note with OCR issues...")
                heatmap = generate_ocr_focused_gradcam(model_predictor.model, image_pil, comprehensive_ocr_results)
                gradcam_filename = f"{uuid.uuid4().hex}_ocr_focused_gradcam.png"
                gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
                Image.fromarray(heatmap).save(gradcam_path)
                explanation = f"Authentic banknote with moderate confidence ({confidence*100:.1f}%). OCR detected some text inconsistencies requiring attention."
            else:
                explanation = f"Authentic banknote with moderate confidence ({confidence*100:.1f}%). OCR analysis shows no text issues."
        
        else:
            # UNCERTAIN case
            result = "UNCERTAIN"
            authenticity_score = confidence if is_real else 1 - confidence
            
            print("Generating OCR-focused GradCAM for UNCERTAIN note...")
            print(f"OCR results for UNCERTAIN note: {len(comprehensive_ocr_results.get('missing_texts', []))} missing texts, {len(comprehensive_ocr_results.get('forbidden_texts_found', []))} forbidden texts")
            
            heatmap = generate_ocr_focused_gradcam(model_predictor.model, image_pil, comprehensive_ocr_results)
            gradcam_filename = f"{uuid.uuid4().hex}_ocr_focused_gradcam.png"
            gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
            Image.fromarray(heatmap).save(gradcam_path)
            explanation = "Uncertain authenticity. Manual verification recommended."
        
        # Prepare response for cases that had OCR analysis
        response = {
            'result': result,
            'confidence': round(confidence * 100, 2),
            'authenticity_score': round(authenticity_score * 100, 2),
            'explanation': explanation,
            'ocr_analysis': clean_ocr_results,
            'gradcam_available': gradcam_filename is not None,
            'gradcam_filename': gradcam_filename,
            'gradcam_type': 'ocr_focused' if gradcam_filename else 'none',
            'timestamp': datetime.now().isoformat(),
            'processing_details': {
                'raw_prediction': prediction_result['raw_prediction'],
                'model_confidence': prediction_result['confidence'],
                'ocr_errors_detected': len(comprehensive_ocr_results.get('missing_texts', [])) + len(comprehensive_ocr_results.get('forbidden_texts_found', [])),
                'error_regions_highlighted': len(comprehensive_ocr_results.get('error_regions', [])),
                'ocr_confidence': clean_ocr_results.get('avg_confidence', 0),
                'text_quality_score': clean_ocr_results.get('quality_score', 0),
                'processing_mode': 'full_analysis'
            }
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in authentication: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/gradcam/<filename>', methods=['GET'])
def get_gradcam_image(filename):
    """Serve GradCAM visualization images"""
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'GradCAM image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving GradCAM image: {e}")
        return jsonify({'error': 'Failed to serve image'}), 500

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information and statistics"""
    return jsonify({
        'model_info': model_predictor.get_model_info(),
        'ocr_capabilities': ocr_validator.get_capabilities(),
        'gradcam_features': ["standard_gradcam", "error_focused_gradcam", "ocr_focused_gradcam"],
        'system_info': {
            'upload_limit_mb': MAX_FILE_SIZE / (1024 * 1024),
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'processing_modes': {
                'cnn_only': 'High confidence real notes (>85%)',
                'full_analysis': 'Fake, uncertain, or low confidence real notes'
            }
        }
    })

@app.route('/validate/ocr', methods=['POST'])
def validate_ocr_only():
    """OCR validation endpoint - Always runs OCR regardless of CNN prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save and process file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # OCR validation - get both comprehensive and clean results
        image_pil = Image.open(file_path)
        comprehensive_ocr_results = ocr_validator.validate_comprehensive_text(image_pil)
        clean_ocr_results = ocr_validator.validate_banknote(file_path)
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'ocr_analysis': clean_ocr_results,
            'comprehensive_analysis': {
                'total_texts_detected': len(comprehensive_ocr_results.get('bounding_boxes', [])),
                'missing_texts_count': len(comprehensive_ocr_results.get('missing_texts', [])),
                'forbidden_texts_count': len(comprehensive_ocr_results.get('forbidden_texts_found', [])),
                'error_regions_count': len(comprehensive_ocr_results.get('error_regions', []))
            },
            'timestamp': datetime.now().isoformat(),
            'note': 'This endpoint always runs OCR analysis regardless of CNN prediction'
        })
        
    except Exception as e:
        logger.error(f"Error in OCR validation: {e}")
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500

@app.route('/test/gradcam', methods=['POST'])
def test_gradcam():
    """Test endpoint for OCR-focused GradCAM generation - Always runs regardless of CNN"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Get OCR results
        image_pil = Image.open(file_path)
        comprehensive_ocr_results = ocr_validator.validate_comprehensive_text(image_pil)
        
        # Generate OCR-focused GradCAM
        print("Testing OCR-focused GradCAM generation...")
        print(f"OCR results: {comprehensive_ocr_results}")
        
        heatmap = generate_ocr_focused_gradcam(model_predictor.model, image_pil, comprehensive_ocr_results)
        gradcam_filename = f"{uuid.uuid4().hex}_test_gradcam.png"
        gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], gradcam_filename)
        Image.fromarray(heatmap).save(gradcam_path)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'status': 'success',
            'gradcam_filename': gradcam_filename,
            'ocr_summary': {
                'missing_texts': len(comprehensive_ocr_results.get('missing_texts', [])),
                'forbidden_texts': len(comprehensive_ocr_results.get('forbidden_texts_found', [])),
                'total_detections': len(comprehensive_ocr_results.get('bounding_boxes', []))
            },
            'message': 'OCR-focused GradCAM generated successfully (test mode)',
            'note': 'This endpoint always runs GradCAM regardless of CNN prediction'
        })
        
    except Exception as e:
        logger.error(f"Error in test GradCAM: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting ScanSure Banknote Authentication API...")
    print("Directory structure created")
    print("Components initialized")
    print(f"\nConfidence thresholds:")
    print(f"  Real note threshold: {model_predictor.confidence_threshold_real*100}%")
    print(f"  Fake note threshold: {model_predictor.confidence_threshold_fake*100}%")
    print("\nProcessing logic:")
    print("  • High confidence REAL notes (>85%): CNN only, no OCR, no GradCAM")
    print("  • All other cases: Full analysis with OCR and GradCAM")
    print("\nAvailable endpoints:")
    print("  POST /authenticate - Authenticate a banknote image (conditional OCR/GradCAM)")
    print("  POST /validate/ocr - OCR validation only (always runs)")
    print("  POST /test/gradcam - Test OCR-focused GradCAM generation (always runs)")
    print("  GET  /gradcam/<filename> - Get GradCAM visualization")
    print("  GET  /health - Health check")
    print("  GET  /model/info - System information")
    
    # 🚀 RENDER-READY CONFIGURATION
    port = int(os.environ.get('PORT', 5000))
    print(f"\nServer running on: http://0.0.0.0:{port}")
    print("✅ Render-ready configuration applied!")
    print("\nNOTE: Main /authenticate endpoint now skips OCR for high-confidence real notes!")
    
    # Use production settings for Render
    app.run(debug=False, host='0.0.0.0', port=port)