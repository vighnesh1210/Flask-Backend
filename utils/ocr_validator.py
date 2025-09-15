import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
from difflib import SequenceMatcher

class OCRValidator:
    def __init__(self):
        # Comprehensive text templates for Indian currency notes
        self.note_templates = {
            "common_required_texts": [
                "Reserve Bank of India",
                "भारतीय रिज़र्व बैंक",
                "India",
                "भारत"
            ],
            "denominations": {
                "10": {
                    "english": ["Ten Rupees", "10"],
                    "hindi": ["दस रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "20": {
                    "english": ["Twenty Rupees", "20"],
                    "hindi": ["बीस रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "50": {
                    "english": ["Fifty Rupees", "50"],
                    "hindi": ["पचास रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "100": {
                    "english": ["One Hundred Rupees", "100"],
                    "hindi": ["एक सौ रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "200": {
                    "english": ["Two Hundred Rupees", "200"],
                    "hindi": ["दो सौ रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "500": {
                    "english": ["Five Hundred Rupees", "500"],
                    "hindi": ["पांच सौ रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                },
                "2000": {
                    "english": ["Two Thousand Rupees", "2000"],
                    "hindi": ["दो हजार रुपए"],
                    "serial_pattern": r"[0-9][A-Z]{2}[0-9]{6}"
                }
            },
            "security_features": [
                "Governor",
                "गवर्नर",
                "Mahatma Gandhi",
                "महात्मा गांधी"
            ],
            "additional_texts": [
                "I promise to pay the bearer",
                "मैं धारक को अदा करने का वचन देता हूँ"
            ],
            "forbidden_texts": [
                "SPECIMEN", "specimen", "Specimen",
                "SAMPLE", "sample", "Sample",
                "NOT LEGAL TENDER",
                "COPY", "copy", "Copy",
                "DUPLICATE", "duplicate", "Duplicate",
                "FOR TRAINING PURPOSE",
                "TRAINING",
                "TEST NOTE",
                "Children ",
                "CHILDREN'S BANK",
                "CHILDRENS BANK",
                "Childrens Bank of India",
            ]
        }
        
        # Test OCR availability
        try:
            pytesseract.get_tesseract_version()
            self.is_available = True
            print("✅ OCR (Tesseract) is available")
        except:
            self.is_available = False
            print("❌ OCR (Tesseract) not found")
    
    def preprocess_for_ocr(self, image):
        """Advanced preprocessing for better OCR results"""
        # Ensure we have a proper image array that owns its data
        if isinstance(image, str):
            image = Image.open(image)
        
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img = np.copy(img)  # Ensure data ownership
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.copy(gray)  # Ensure data ownership
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced = np.copy(enhanced)  # Ensure data ownership
        
        denoised = cv2.fastNlMeansDenoising(enhanced)
        denoised = np.copy(denoised)  # Ensure data ownership
        
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = np.copy(thresh)  # Ensure data ownership
        
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = np.copy(processed)  # Ensure data ownership
        
        return processed
    
    def extract_comprehensive_text(self, image):
        """Extract text with multiple OCR configurations"""
        processed_image = self.preprocess_for_ocr(image)
        all_text = ""
        bounding_boxes = []
        ocr_configs = ['--psm 6', '--psm 3', '--psm 8', '--psm 11']
        
        for config in ocr_configs:
            try:
                data = pytesseract.image_to_data(
                    processed_image, 
                    output_type=pytesseract.Output.DICT,
                    config=config
                )
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    if text and conf > 30:
                        all_text += text + " "
                        bounding_boxes.append({
                            'text': text,
                            'bbox': [data['left'][i], data['top'][i], data['width'][i], data['height'][i]],
                            'confidence': conf
                        })
            except Exception as e:
                print(f"OCR config {config} failed: {e}")
                continue
        return all_text.strip(), bounding_boxes
    
    def detect_denomination(self, extracted_text):
        text_lower = extracted_text.lower()
        for denom, patterns in self.note_templates["denominations"].items():
            for eng_text in patterns["english"]:
                if eng_text.lower() in text_lower:
                    return denom
        return None
    
    def similarity_check(self, text1, text2, threshold=0.8):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() >= threshold
    
    def validate_comprehensive_text(self, image):
        """Comprehensive text validation with checks"""
        if not self.is_available:
            return {"extracted_text": "", "validation_results": [], "is_valid": False, "error": "OCR not available"}
        try:
            extracted_text, bounding_boxes = self.extract_comprehensive_text(image)
            detected_denomination = self.detect_denomination(extracted_text)
            validation_results = []
            missing_texts, spelling_errors, error_regions, forbidden_found = [], [], [], []
            
            # Forbidden text check
            for forbidden_text in self.note_templates["forbidden_texts"]:
                if forbidden_text.lower() in extracted_text.lower():
                    for bbox_info in bounding_boxes:
                        if forbidden_text.lower() in bbox_info['text'].lower():
                            forbidden_found.append({
                                "forbidden_text": forbidden_text,
                                "found_text": bbox_info['text'],
                                "bbox": bbox_info['bbox'],
                                "confidence": bbox_info['confidence']
                            })
                            error_regions.append(bbox_info['bbox'])
                            validation_results.append({
                                "text": forbidden_text,
                                "status": "FORBIDDEN_FOUND",
                                "type": "critical_security_violation",
                                "found_as": bbox_info['text']
                            })
                            break
            if forbidden_found:
                return {
                    "extracted_text": extracted_text,
                    "detected_denomination": detected_denomination,
                    "validation_results": validation_results,
                    "forbidden_texts_found": forbidden_found,
                    "error_regions": error_regions,
                    "bounding_boxes": bounding_boxes,
                    "is_valid": False,
                    "is_specimen": True,
                    "critical_security_violation": True
                }
            
            # Required text check
            for required_text in self.note_templates["common_required_texts"]:
                if required_text.lower() in extracted_text.lower():
                    validation_results.append({"text": required_text, "status": "found", "type": "required"})
                else:
                    missing_texts.append(required_text)
                    validation_results.append({"text": required_text, "status": "missing", "type": "required"})
            
            # Security features
            for security_text in self.note_templates["security_features"]:
                if security_text.lower() in extracted_text.lower():
                    validation_results.append({"text": security_text, "status": "found", "type": "security"})
                else:
                    missing_texts.append(security_text)
                    validation_results.append({"text": security_text, "status": "missing", "type": "security"})
            
            is_valid = len(missing_texts) == 0 and len(spelling_errors) == 0
            return {
                "extracted_text": extracted_text,
                "detected_denomination": detected_denomination,
                "validation_results": validation_results,
                "missing_texts": missing_texts,
                "spelling_errors": spelling_errors,
                "forbidden_texts_found": forbidden_found,
                "error_regions": error_regions,
                "bounding_boxes": bounding_boxes,
                "is_valid": is_valid,
                "is_specimen": False
            }
        except Exception as e:
            return {"extracted_text": "", "validation_results": [], "is_valid": False, "error": str(e)}

    def validate_banknote(self, image_path):
        """Main method called by app.py - validates banknote using OCR with clean output"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            # Perform comprehensive text validation
            results = self.validate_comprehensive_text(image)
            
            # If there's an error, return it
            if results.get('error'):
                return {
                    'status': 'error',
                    'message': results['error']
                }
            
            # Check for critical security violations first
            if results.get('is_specimen') or results.get('forbidden_texts_found'):
                return {
                    'status': 'counterfeit',
                    'reason': 'Forbidden text detected',
                    'forbidden_texts': [item['forbidden_text'] for item in results.get('forbidden_texts_found', [])]
                }
            
            # Extract only the problems/issues
            issues = {
                'missing_required_texts': [],
                'missing_security_features': [],
                'denomination_issues': []
            }
            
            # Process validation results to find only missing items
            for validation in results.get('validation_results', []):
                if validation['status'] == 'missing':
                    if validation['type'] == 'required':
                        issues['missing_required_texts'].append(validation['text'])
                    elif validation['type'] == 'security':
                        issues['missing_security_features'].append(validation['text'])
            
            # Check denomination detection
            if not results.get('detected_denomination'):
                issues['denomination_issues'].append('Could not detect denomination')
            
            # Clean output - only return issues if they exist
            clean_result = {}
            
            if issues['missing_required_texts']:
                clean_result['missing_required_texts'] = issues['missing_required_texts']
            
            if issues['missing_security_features']:
                clean_result['missing_security_features'] = issues['missing_security_features']
                
            if issues['denomination_issues']:
                clean_result['denomination_issues'] = issues['denomination_issues']
            
            # Determine overall status
            if any(issues.values()):
                clean_result['status'] = 'issues_found'
                clean_result['is_valid'] = False
            else:
                clean_result['status'] = 'valid'
                clean_result['is_valid'] = True
            
            # Optionally include detected denomination if found
            if results.get('detected_denomination'):
                clean_result['detected_denomination'] = results.get('detected_denomination')
            
            return clean_result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_capabilities(self):
        return {
            "ocr_available": self.is_available,
            "supported_languages": ["eng", "hin"],
            "checks": [
                "denomination detection",
                "required text validation",
                "forbidden text detection",
                "spelling similarity check"
            ],
            "description": "Validates Indian currency notes by OCR text analysis"
        }