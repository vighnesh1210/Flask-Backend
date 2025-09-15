import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import logging

class ModelPredictor:
    def __init__(self, model_path):
        # Initialize attributes FIRST to prevent AttributeError
        self.is_loaded = False
        self.model = None
        self.confidence_threshold_real = 0.85
        self.confidence_threshold_fake = 0.85
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Try to load model
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Internal method to load model with multiple fallback strategies"""
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model file not found: {model_path}")
            return
        
        try:
            # Strategy 1: Standard loading
            self.model = load_model(model_path)
            
            # Build the model by running dummy data through it
            dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)

            self.is_loaded = True
            self.logger.info(f"✅ Model loaded successfully from {model_path}")
            self.logger.info(f"📊 Input shape: {self.model.input_shape}")
            self.logger.info(f"📊 Output shape: {self.model.output_shape}")

            # Print device info
            self._print_device_info()

        except Exception as e:
            self.logger.error(f"❌ Standard model loading failed: {e}")
            
            # Strategy 2: Load without compilation
            try:
                self.logger.info("🔄 Trying to load model without compilation...")
                self.model = load_model(model_path, compile=False)
                
                # Recompile the model
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Test with dummy data
                dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
                self.model.predict(dummy_input, verbose=0)
                
                self.is_loaded = True
                self.logger.info("✅ Model loaded successfully without compilation")
                self._print_device_info()
                
            except Exception as e2:
                self.logger.error(f"❌ All model loading strategies failed: {e2}")
                self.logger.warning("🔄 Running in fallback mode - predictions will return dummy results")
    
    def _print_device_info(self):
        """Print available device information"""
        try:
            devices = tf.config.list_physical_devices()
            self.logger.info("💻 Available devices:")
            for d in devices:
                self.logger.info(f"   - {d.device_type}: {d.name}")
            
            if tf.config.list_physical_devices("GPU"):
                self.logger.info("⚡ Running with GPU acceleration")
            else:
                self.logger.info("🖥️ Running on CPU")
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
    
    def is_model_loaded(self):
        """Check if model was loaded properly"""
        return self.is_loaded

    def preprocess_image(self, image, target_size=(150, 150)):
        """Preprocess image for model prediction"""
        try:
            if isinstance(image, str):  
                image = Image.open(image)  # if path is passed, open image
            
            # Resize image
            image_resized = image.resize(target_size)
            
            # Convert to numpy array and ensure it owns its data
            img_array = np.array(image_resized)
            img_array = np.copy(img_array)  # Fix: Ensure array owns its data
            
            # Normalize pixels to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image):
        """Predict if currency note is real or fake"""
        if not self.is_loaded:
            # Return fallback response when model isn't loaded
            self.logger.warning("Model not loaded - returning fallback prediction")
            return {
                "is_real": None,
                "confidence": 0.0,
                "raw_prediction": 0.5,
                "error": "Model not loaded",
                "fallback_mode": True
            }
        
        try:
            # Check if image is already preprocessed (from ImageProcessor)
            if isinstance(image, np.ndarray) and len(image.shape) == 4:
                # Image is already preprocessed by ImageProcessor
                processed_image = np.copy(image)  # Ensure data ownership
                
                # Check if we need to resize for model input (150x150 vs 224x224)
                if processed_image.shape[1:3] != (150, 150):
                    # Need to resize from ImageProcessor size (224x224) to model size (150x150)
                    from PIL import Image as PILImage
                    
                    # Convert back to PIL for resizing
                    img_for_resize = (processed_image[0] * 255).astype(np.uint8)
                    pil_image = PILImage.fromarray(img_for_resize)
                    
                    # Resize to model input size
                    resized_pil = pil_image.resize((150, 150))
                    
                    # Convert back to model input format
                    processed_image = np.array(resized_pil, dtype=np.float32) / 255.0
                    processed_image = np.expand_dims(processed_image, axis=0)
            else:
                # Image needs preprocessing
                processed_image = self.preprocess_image(image)
            
            # Get prediction
            prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
            
            # Convert to binary classification
            if prediction_prob > 0.5:
                is_real = True
                confidence = float(prediction_prob)
            else:
                is_real = False
                confidence = float(1 - prediction_prob)
            
            return {
                "is_real": is_real,
                "confidence": confidence,
                "raw_prediction": float(prediction_prob),
                "fallback_mode": False
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                "is_real": None,
                "confidence": 0.0,
                "raw_prediction": 0.5,
                "error": str(e),
                "fallback_mode": True
            }
    
    def get_model_info(self):
        """Return model info for /model/info endpoint"""
        if not self.is_loaded:
            return {
                "status": "Model not loaded",
                "fallback_mode": True,
                "confidence_threshold_real": self.confidence_threshold_real,
                "confidence_threshold_fake": self.confidence_threshold_fake
            }
        
        try:
            return {
                "status": "Model loaded successfully",
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "trainable_params": int(np.sum([np.prod(v.shape) for v in self.model.trainable_weights])),
                "non_trainable_params": int(np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])),
                "confidence_threshold_real": self.confidence_threshold_real,
                "confidence_threshold_fake": self.confidence_threshold_fake,
                "fallback_mode": False
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {
                "status": f"Error: {str(e)}",
                "fallback_mode": True
            }