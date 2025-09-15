import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from PIL import Image


class OCRFocusedGradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = None
        
        print("🎯 Initializing OCR-Focused GradCAM...")
        
        # Build the model first
        try:
            dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
            _ = self.model(dummy_input)
            self.target_layer = self._find_target_layer()
            print(f"✅ Target layer selected: {self.target_layer}")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")

    def _find_target_layer(self):
        """Find the last convolutional layer (handles nested Sequential)"""
        conv_layer = None

        def search_layers(layers):
            nonlocal conv_layer
            for layer in reversed(layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    conv_layer = layer.name
                    return
                if isinstance(layer, tf.keras.Sequential):
                    search_layers(layer.layers)

        search_layers(self.model.layers)
        return conv_layer or self.model.layers[-1].name

    def _preprocess_image(self, image, target_size):
        """Preprocess image for model input"""
        img_array = np.array(image.resize(target_size))
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def generate_ocr_focused_heatmap(self, image, ocr_results, target_size=(150, 150)):
        """Generate GradCAM focused ONLY on OCR error regions"""
        print("🔍 Generating OCR-focused heatmap...")

        error_regions = self._extract_error_regions(ocr_results, image.size, target_size)
        
        if not error_regions:
            print("ℹ️ No OCR errors detected, returning black image")
            return self._create_black_background(target_size)
        
        print(f"🎯 Found {len(error_regions)} error regions to focus on")

        base_heatmap = self._generate_base_gradcam(image, target_size)

        focused_heatmap = self._create_focused_visualization(
            image, base_heatmap, error_regions, target_size
        )
        
        return focused_heatmap

    def _extract_error_regions(self, ocr_results, original_size, target_size):
        """Extract error regions from OCR validation results"""
        error_regions = []
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        if ocr_results.get('forbidden_texts_found'):
            for forbidden_item in ocr_results['forbidden_texts_found']:
                if 'bbox' in forbidden_item:
                    x, y, w, h = forbidden_item['bbox']
                    scaled_region = {
                        'x': int(x * scale_x),
                        'y': int(y * scale_y),
                        'w': int(w * scale_x),
                        'h': int(h * scale_y),
                        'type': 'forbidden',
                        'text': forbidden_item.get('forbidden_text', 'unknown')
                    }
                    error_regions.append(scaled_region)
                    print(f"🚫 Forbidden text region: {scaled_region}")

        if ocr_results.get('bounding_boxes'):
            for bbox_info in ocr_results['bounding_boxes']:
                if bbox_info['confidence'] < 50:
                    x, y, w, h = bbox_info['bbox']
                    scaled_region = {
                        'x': int(x * scale_x),
                        'y': int(y * scale_y),
                        'w': int(w * scale_x),
                        'h': int(h * scale_y),
                        'type': 'low_confidence',
                        'text': bbox_info['text']
                    }
                    error_regions.append(scaled_region)
                    print(f"❓ Low confidence region: {scaled_region}")

        if not error_regions and ocr_results.get('missing_texts'):
            common_areas = [
                {'x': int(0.1 * target_size[0]), 'y': int(0.1 * target_size[1]), 
                 'w': int(0.3 * target_size[0]), 'h': int(0.2 * target_size[1]), 
                 'type': 'missing', 'text': 'header_area'},
                {'x': int(0.6 * target_size[0]), 'y': int(0.1 * target_size[1]), 
                 'w': int(0.3 * target_size[0]), 'h': int(0.2 * target_size[1]), 
                 'type': 'missing', 'text': 'corner_area'},
                {'x': int(0.2 * target_size[0]), 'y': int(0.7 * target_size[1]), 
                 'w': int(0.6 * target_size[0]), 'h': int(0.2 * target_size[1]), 
                 'type': 'missing', 'text': 'footer_area'}
            ]
            error_regions.extend(common_areas)
            print("📍 Using common text areas for missing text focus")

        return error_regions

    def _generate_base_gradcam(self, image, target_size):
        """Generate the base GradCAM heatmap"""
        try:
            img_array = self._preprocess_image(image, target_size)
            target_layer_obj = self.model.get_layer(self.target_layer)

            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[target_layer_obj.output, self.model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                    class_score = predictions[0, 0]
                else:
                    class_score = tf.reduce_max(predictions[0])

            grads = tape.gradient(class_score, conv_outputs)

            if grads is None:
                print("⚠️ No gradients computed, using activation-based heatmap")
                return self._create_activation_heatmap(conv_outputs[0], target_size)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            weighted_map = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
            for i in range(conv_outputs.shape[-1]):
                weighted_map += conv_outputs[:, :, i].numpy() * pooled_grads[i].numpy()

            heatmap = np.maximum(weighted_map, 0)
            heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
            return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)

        except Exception as e:
            print(f"❌ Error generating base GradCAM: {e}")
            return np.random.random(target_size) * 0.3

    def _create_activation_heatmap(self, conv_outputs, target_size):
        """Create heatmap from activations when gradients fail"""
        activation_map = tf.reduce_max(conv_outputs, axis=-1)
        activation_map = activation_map - tf.reduce_min(activation_map)
        activation_map = activation_map / (tf.reduce_max(activation_map) + 1e-8)
        return cv2.resize(activation_map.numpy(), target_size, interpolation=cv2.INTER_CUBIC)

    def _create_focused_visualization(self, image, base_heatmap, error_regions, target_size):
        """Overlay Grad-CAM ONLY on error regions"""
        original_resized = np.array(image.resize(target_size))
        if len(original_resized.shape) == 3 and original_resized.shape[2] == 4:
            original_resized = original_resized[:, :, :3]

        error_mask = np.zeros(target_size, dtype=np.float32)

        for region in error_regions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            x_end, y_end = min(x + w, target_size[0]), min(y + h, target_size[1])

            if x_end > x and y_end > y:
                error_mask[y:y_end, x:x_end] = 1.0

        error_mask = cv2.GaussianBlur(error_mask, (15, 15), 0)
        focused_heatmap = base_heatmap * error_mask

        if np.max(focused_heatmap) > 0:
            focused_heatmap = focused_heatmap / np.max(focused_heatmap)

        return self._create_final_visualization(original_resized, focused_heatmap, target_size)

    def _create_final_visualization(self, original_image, focused_heatmap, target_size):
        heatmap_255 = np.uint8(255 * np.clip(focused_heatmap, 0, 1))
        heatmap_colored = cv2.applyColorMap(heatmap_255, cv2.COLORMAP_HOT)

        if len(original_image.shape) == 3:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        result = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def _create_black_background(self, target_size):
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)


# Wrapper functions for app.py
def generate_ocr_focused_gradcam(model, image, ocr_results):
    try:
        gradcam = OCRFocusedGradCAM(model)
        return gradcam.generate_ocr_focused_heatmap(image, ocr_results)
    except Exception as e:
        print(f"❌ Error generating OCR-focused GradCAM: {e}")
        return np.zeros((150, 150, 3), dtype=np.uint8)

def generate_gradcam(model, image, target_size=(150, 150)):
    try:
        gradcam = OCRFocusedGradCAM(model)
        return gradcam._generate_base_gradcam(image, target_size)
    except:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

def generate_error_focused_gradcam(model, image, error_regions):
    try:
        ocr_results = {
            'forbidden_texts_found': [],
            'bounding_boxes': [{'bbox': bbox, 'confidence': 30, 'text': 'error_region'} for bbox in error_regions],
            'missing_texts': []
        }
        return generate_ocr_focused_gradcam(model, image, ocr_results)
    except:
        return np.zeros((150, 150, 3), dtype=np.uint8)
