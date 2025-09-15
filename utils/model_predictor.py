import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image


class ModelPredictor:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path)
            
            # Build the model by running dummy data through it
            dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
            self.model.predict(dummy_input)

            self.is_loaded = True
            self.confidence_threshold_real = 0.85  # tweak as per your dataset
            self.confidence_threshold_fake = 0.85  # tweak as per your dataset

            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Input shape: {self.model.input_shape}")
            print(f"📊 Output shape: {self.model.output_shape}")

            # Print device info
            devices = tf.config.list_physical_devices()
            print("💻 Available devices:")
            for d in devices:
                print(f"   - {d.device_type}: {d.name}")
            if tf.config.list_physical_devices("GPU"):
                print("⚡ Running with GPU acceleration")
            else:
                print("🖥️ Running on CPU")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
            self.model = None
    
    def is_model_loaded(self):
        """Check if model was loaded properly"""
        return self.is_loaded

    def preprocess_image(self, image, target_size=(150, 150)):
        """Preprocess image for model prediction"""
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
    
    def predict(self, image):
        """Predict if currency note is real or fake"""
        if not self.is_loaded:
            raise Exception("Model not loaded")
        
        # Check if image is already preprocessed (from ImageProcessor)
        if isinstance(image, np.ndarray) and len(image.shape) == 4:
            # Image is already preprocessed by ImageProcessor
            # But ensure it owns its data and has correct target size for model
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
        prediction_prob = self.model.predict(processed_image)[0][0]
        
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
            "raw_prediction": float(prediction_prob)
        }
    
    def get_model_info(self):
        """Return model info for /model/info endpoint"""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return {
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "trainable_params": int(np.sum([np.prod(v.shape) for v in self.model.trainable_weights])),
            "non_trainable_params": int(np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights]))
        }
