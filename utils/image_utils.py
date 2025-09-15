import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.img_height, self.img_width = target_size

    def _load_image(self, image_input):
        """Helper: load image from path, bytes, or numpy array"""
        if isinstance(image_input, np.ndarray):
            return np.ascontiguousarray(image_input)

        if isinstance(image_input, bytes):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return np.ascontiguousarray(img) if img is not None else None

        if isinstance(image_input, str):  # path
            img = cv2.imread(image_input)
            return np.ascontiguousarray(img) if img is not None else None

        logger.error("Unsupported image input type")
        return None

    def preprocess_for_model(self, image_input):
        """Preprocess image for CNN model input"""
        try:
            img = self._load_image(image_input)
            if img is None:
                raise ValueError("Could not load image")

            # Ensure contiguous array that owns its data
            img = np.copy(img)

            # Convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize safely and ensure data ownership
            img_resized = cv2.resize(img, (self.img_width, self.img_height))
            img_resized = np.copy(img_resized)  # Fix: Ensure resized array owns its data

            # Normalize
            img_normalized = img_resized.astype(np.float32) / 255.0

            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            return img_batch

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def preprocess_for_ocr(self, image_input):
        """Preprocess image for better OCR results"""
        try:
            img = self._load_image(image_input)
            if img is None:
                return None

            # Ensure contiguous array that owns its data
            img = np.copy(img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.copy(gray)  # Ensure data ownership
            
            denoised = cv2.fastNlMeansDenoising(gray)
            denoised = np.copy(denoised)  # Ensure data ownership
            
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=30)
            enhanced = np.copy(enhanced)  # Ensure data ownership

            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            thresh = np.copy(thresh)  # Ensure data ownership

            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = np.copy(cleaned)  # Fix: Ensure final array owns its data
            
            return cleaned

        except Exception as e:
            logger.error(f"Error preprocessing image for OCR: {e}")
            return None

    def enhance_image(self, image_input, brightness=1.1, contrast=1.2, sharpness=1.1):
        """Enhance image quality"""
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                img = image_input

            # Apply enhancements
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)

            return img

        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return None

    def detect_edges(self, image_input, low_threshold=50, high_threshold=150):
        """Detect edges in the image"""
        try:
            img = self._load_image(image_input)
            if img is None:
                return None

            # Ensure data ownership
            img = np.copy(img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.copy(gray)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            blurred = np.copy(blurred)
            
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            edges = np.copy(edges)  # Ensure final array owns its data
            
            return edges

        except Exception as e:
            logger.error(f"Error detecting edges: {e}")
            return None