from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from PIL import Image

# Set environment variables to reduce TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

print("Starting ScanSure Banknote Authentication API...")
print("Directory structure created")

# Initialize components with error handling
try:
    image_processor = ImageProcessor()
    logger.info("✅ ImageProcessor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ImageProcessor: {e}")
    image_processor = None

# Initialize model predictor with proper path handling
model_path = os.path.join(MODEL_FOLDER, 'fake_note_detector_final.h5')
if not os.path.exists(model_path):
    # Try alternative paths
    alternative_paths = [
        'fake_note_detector_final.h5',
        'banknote_model.h5',
        'model.h5'
    ]
    
    model_path = None
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            model_path = alt_path
            logger.info(f"Found model at alternative path: {alt_path}")
            break
        # Also check in MODEL_FOLDER
        full_alt_path = os.path.join(MODEL_FOLDER, alt_path)
        if os.path.exists(full_alt_path):
            model_path = full_alt_path
            logger.info(f"Found model at: {full_alt_path}")
            break
    
    if model_path is None:
        logger.warning("❌ No model file found. Available files:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.h5', '.keras')):
                    logger.warning(f"  Found model file: {os.path.join(root, file)}")

try:
    model_predictor = ModelPredictor(model_path) if model_path else ModelPredictor("")
    logger.info("✅ ModelPredictor initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize ModelPredictor: {e}")
    # Create a fallback ModelPredictor
    class FallbackModelPredictor:
        def __init__(self):
            self.confidence_threshold_real = 0.85
            self.confidence_threshold_fake = 0.85
            self.is_loaded = False
            
        def is_model_loaded(self):
            return False
            
        def predict(self, image):
            return {
                "is_real": None,
                "confidence": 0.0,
                "raw_prediction": 0.5,
                "error": "Model not loaded",
                "fallback_mode": True
            }
            
        def get_model_info(self):
            return {
                "status": "Model not loaded - using fallback mode",
                "fallback_mode": True
            }
    
    model_predictor = FallbackModelPredictor()

try:
    ocr_validator = OCRValidator()
    logger.info("✅ OCRValidator initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize OCRValidator: {e}")
    # Create a fallback OCRValidator
    class FallbackOCRValidator:
        def validate_comprehensive_text(self, image):
            return {
                'missing_texts': [],
                'forbidden_texts_found': [],
                'error_regions': [],
                'bounding_boxes': []
            }
        
        def validate_banknote(self, file_path):
            return {
                'status': 'OCR not available',
                'avg_confidence': 0,
                'quality_score': 0,
                'error': 'OCR system not available'
            }
        
        def get_capabilities(self):
            return {
                'status': 'OCR not available',
                'tesseract_available': False
            }
    
    ocr_validator = FallbackOCRValidator()

print("Components initialized")

# Add model summary and debugging info - ONLY if model loaded successfully
if model_predictor.is_model_loaded():
    try:
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
    except Exception as e:
        logger.error(f"Error displaying model summary: {e}")
        print("Model loaded but couldn't display summary")
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
            'image_processor': image_processor is not None,
            'ocr_validator': True,  # Always true even if fallback
            'gradcam_available': model_predictor.is_model_loaded()
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
        if image_processor is None:
            return jsonify({'error': 'Image processor not available'}), 500
            
        processed_image = image_processor.preprocess_for_model(file_path)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Get model prediction FIRST
        prediction_result = model_predictor.predict(processed_image)
        
        # Handle fallback mode
        if prediction_result.get('fallback_mode', False):
            response = {
                'result': 'UNCERTAIN',
                'confidence': 0.0,
                'authenticity_score': 0.0,
                'explanation': 'Model not available - cannot perform authentication',
                'ocr_analysis': {
                    'status': 'skipped',
                    'reason': 'Model not loaded'
                },
                'gradcam_available': False,
                'gradcam_filename': None,
                'gradcam_type': 'none',
                'timestamp': datetime.now().isoformat(),
                'processing_details': {
                    'model_confidence': 0.0,
                    'processing_mode': 'fallback',
                    'error': prediction_result.get('error', 'Unknown error')
                }
            }
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify(response)
        
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
        
        # Only generate GradCAM if model is loaded
        if model_predictor.is_model_loaded():
            try:
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
            
            except Exception as gradcam_error:
                logger.error(f"GradCAM generation failed: {gradcam_error}")
                # Continue without GradCAM
                if not is_real and confidence >= model_predictor.confidence_threshold_fake:
                    result = "FAKE"
                    authenticity_score = 1 - confidence
                    explanation = "Counterfeit banknote detected. GradCAM visualization unavailable."
                elif is_real and confidence < model_predictor.confidence_threshold_real:
                    result = "REAL"
                    authenticity_score = confidence
                    explanation = f"Authentic banknote with moderate confidence ({confidence*100:.1f}%). GradCAM visualization unavailable."
                else:
                    result = "UNCERTAIN"
                    authenticity_score = confidence if is_real else 1 - confidence
                    explanation = "Uncertain authenticity. Manual verification recommended. GradCAM visualization unavailable."
        else:
            # Model not loaded, use basic classification
            if confidence > 0.7:
                result = "REAL" if is_real else "FAKE"
                authenticity_score = confidence if is_real else 1 - confidence
                explanation = f"Classification based on fallback analysis. Model not fully available."
            else:
                result = "UNCERTAIN"
                authenticity_score = 0.5
                explanation = "Uncertain authenticity. Model not available for confident prediction."
        
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
                'raw_prediction': prediction_result.get('raw_prediction', 0.5),
                'model_confidence': prediction_result.get('confidence', 0.0),
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
        if not model_predictor.is_model_loaded():
            return jsonify({'error': 'Model not loaded - cannot generate GradCAM'}), 500
        
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
    # Print configuration info ONLY after successful initialization
    print(f"\nConfidence thresholds:")
    if hasattr(model_predictor, 'confidence_threshold_real'):
        print(f"  Real note threshold: {model_predictor.confidence_threshold_real*100}%")
        print(f"  Fake note threshold: {model_predictor.confidence_threshold_fake*100}%")
    else:
        print("  Thresholds not available (model not loaded)")
    
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
    
    # 🚀 RAILWAY-READY CONFIGURATION
    port = int(os.environ.get('PORT', 5000))
    print(f"\nServer running on: http://0.0.0.0:{port}")
    print("✅ Railway-ready configuration applied!")
    print("\nNOTE: Main /authenticate endpoint now skips OCR for high-confidence real notes!")
    
    # Use production settings for Railway
    app.run(debug=False, host='0.0.0.0', port=port)