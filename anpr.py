import cv2
import numpy as np
import easyocr
import os
from datetime import datetime
import pandas as pd
import re
import torch
from PIL import Image, ImageEnhance, ImageFilter

# Import pytesseract conditionally to avoid errors if not installed
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class ANPR:
    def __init__(self):
        # Initialize YOLO model for plate detection
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov5nu.pt')  # Use improved YOLOv5 model
        except:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Fall back to YOLOv8
            
        # Initialize EasyOCR reader with only English for better accuracy
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), 
                                    recog_network='english_g2')
        
        # Try to set up Tesseract as a backup OCR engine
        self.tesseract_available = TESSERACT_AVAILABLE
        if self.tesseract_available:
            try:
                # Check if tesseract is available
                pytesseract.get_tesseract_version()
                # Configure tesseract for license plates (PSM 7 - single line of text)
                self.custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            except:
                self.tesseract_available = False
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame(columns=['filename', 'plate_number', 'confidence', 'timestamp'])
        
        # Load the model with a small dummy image
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8))
        
        # Indian state codes pattern for validation
        self.indian_state_codes = [
            'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
            'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 
            'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 
            'TS', 'UK', 'UP', 'WB'
        ]

    def detect_plates(self, frame):
        """Detect number plates in a frame using YOLO with improved parameters for Indian plates"""
        # Resize image for better detection while maintaining aspect ratio
        height, width = frame.shape[:2]
        
        # Ensure minimum dimensions for detection
        if width < 640 or height < 640:
            scale = max(640 / width, 640 / height)
            frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        else:
            frame_resized = frame.copy()
            
        # Run YOLO detection with lower confidence threshold for Indian plates
        # and higher IoU threshold for better detection
        results = self.model(frame_resized, conf=0.15, iou=0.45, classes=[0, 2, 3, 5, 7])  # Focus on common objects where plates might be found
        plates = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # If the image was resized, adjust coordinates back to original scale
                if frame_resized is not frame:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Add more padding around the detection to ensure the plate is fully captured
                # Increased padding for Indian plates which may have state emblems
                padding_x = int((x2 - x1) * 0.15)  # 15% padding
                padding_y = int((y2 - y1) * 0.20)  # 20% padding
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                # Calculate aspect ratio to filter out unlikely plate shapes
                aspect_ratio = (x2 - x1) / float(y2 - y1)
                
                # Indian license plates typically have aspect ratios between 2.0 and 5.0
                if 1.5 <= aspect_ratio <= 5.5:
                    plates.append((x1, y1, x2, y2))
        
        # If no plates detected with YOLO, try using edge detection as fallback
        if not plates:
            plates = self._detect_plates_using_edges(frame)
            
        return plates

    def _detect_plates_using_edges(self, frame):
        """Fallback method to detect license plates using edge detection - optimized for Indian plates"""
        plates = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Find edges using Canny edge detector with parameters tuned for Indian plates
        edged = cv2.Canny(gray, 30, 200)
        
        # Dilate the edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        # Find contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]  # Increased from 15 to 20
        
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If the contour has 4-6 points, it's likely a rectangle (license plate)
            if len(approx) >= 4 and len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio typical for Indian license plates
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 5.5:
                    # Add padding to ensure full plate is captured
                    padding_x = int(w * 0.10)
                    padding_y = int(h * 0.10)
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w = min(frame.shape[1] - x, w + 2 * padding_x)
                    h = min(frame.shape[0] - y, h + 2 * padding_y)
                    plates.append((x, y, x + w, y + h))
        
        return plates
    
    def _enhance_plate_image(self, plate_img):
        """Apply multiple image enhancement techniques to improve OCR accuracy"""
        enhanced_images = []
        
        # Convert to PIL Image for easier enhancement
        if isinstance(plate_img, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = plate_img
        
        # Resize image to improve OCR (larger is better for OCR)
        width, height = pil_img.size
        scale_factor = 2.0  # Double the size
        resized_img = pil_img.resize((int(width * scale_factor), int(height * scale_factor)), 
                                     Image.Resampling.LANCZOS)
        enhanced_images.append(cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR))
            
        # Original image
        enhanced_images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(resized_img)
        contrast_img = contrast_enhancer.enhance(2.0)  # Increase contrast
        enhanced_images.append(cv2.cvtColor(np.array(contrast_img), cv2.COLOR_RGB2BGR))
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(resized_img)
        sharp_img = sharpness_enhancer.enhance(2.0)  # Increase sharpness
        enhanced_images.append(cv2.cvtColor(np.array(sharp_img), cv2.COLOR_RGB2BGR))
        
        # Apply median filter to reduce noise
        median_img = resized_img.filter(ImageFilter.MedianFilter(size=3))
        enhanced_images.append(cv2.cvtColor(np.array(median_img), cv2.COLOR_RGB2BGR))
        
        # Apply unsharp mask filter to enhance edges
        unsharp_img = resized_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        enhanced_images.append(cv2.cvtColor(np.array(unsharp_img), cv2.COLOR_RGB2BGR))
        
        # Convert to OpenCV format and apply additional processing
        for img in enhanced_images.copy():
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced_images.append(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            enhanced_images.append(thresh)
            
            # Apply Otsu's thresholding
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_images.append(otsu)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            enhanced_images.append(morph_close)
            
            # Apply morphological operations to the Otsu result
            morph_open = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
            enhanced_images.append(morph_open)
            
            # Add inverted images (white text on black background)
            inverted = cv2.bitwise_not(gray)
            enhanced_images.append(inverted)
            
            # Try different binarization thresholds
            for threshold in [100, 120, 140, 160, 180]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                enhanced_images.append(binary)
        
        return enhanced_images

    def extract_text(self, frame, plate_coords):
        """Extract text from detected plate region with improved preprocessing for Indian plates"""
        x1, y1, x2, y2 = plate_coords
        # Ensure coordinates are within frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 10:
            return None, 0.0
            
        plate_img = frame[y1:y2, x1:x2]
        
        # Apply multiple enhancement techniques
        enhanced_images = self._enhance_plate_image(plate_img)
        
        # Try OCR with EasyOCR on all enhanced images
        all_results = []
        
        for img in enhanced_images:
            try:
                # Use allowlist to restrict to characters found on license plates
                results = self.reader.readtext(img, detail=1, paragraph=False, 
                                              allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                
                # Filter out results with very low confidence or short text
                filtered_results = [r for r in results if r[2] > 0.3 and len(r[1]) > 2]
                all_results.extend(filtered_results)
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        # Try with Tesseract as well if available
        if self.tesseract_available:
            for img in enhanced_images:
                try:
                    # Use Tesseract with optimized config for license plates
                    text = pytesseract.image_to_string(img, config=self.custom_config)
                    if text and len(text.strip()) > 2:
                        # Add to results with a moderate confidence
                        all_results.append(([0, 0, 0, 0], text.strip(), 0.5))
                except Exception as e:
                    print(f"Tesseract error: {e}")
        
        if all_results:
            # Sort results by confidence
            all_results.sort(key=lambda x: x[2], reverse=True)
            
            # Try to find the best result that looks like an Indian license plate
            for result in all_results:
                text = result[1]
                confidence = result[2]
                
                # Clean up the text (remove spaces and special characters)
                text = ''.join(c for c in text if c.isalnum() or c == '-')
                
                # Format according to Indian license plate pattern (if possible)
                formatted_text = self._format_indian_plate(text)
                
                # Check if the text matches Indian license plate pattern
                if self._is_likely_indian_plate(formatted_text):
                    return formatted_text, confidence
            
            # If no good match found, return the highest confidence result
            best_result = all_results[0]
            text = best_result[1]
            confidence = best_result[2]
            
            # Clean up the text
            text = ''.join(c for c in text if c.isalnum() or c == '-')
            
            # Format according to Indian license plate pattern
            formatted_text = self._format_indian_plate(text)
            return formatted_text, confidence
                
        return None, 0.0
    
    def _is_likely_indian_plate(self, text):
        """Check if the text is likely to be an Indian license plate"""
        if not text or len(text) < 4:
            return False
            
        # Check for common patterns in Indian plates
        text = text.upper()
        
        # Check if first two characters are a valid state code
        if len(text) >= 2 and text[:2] in self.indian_state_codes:
            return True
            
        # Check for digits and letters in the right places
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', text.replace('-', '')):
            return True
            
        return False
        
    def _format_indian_plate(self, text):
        """Format text according to Indian license plate pattern if possible"""
        # Remove all spaces and special characters
        text = ''.join(c for c in text if c.isalnum())
        
        # Convert to uppercase
        text = text.upper()
        
        # Check if it looks like "IND" or other generic text
        if text == "IND" or len(text) < 4:
            return text
            
        # Try to match Indian license plate pattern: [State Code][District Code] [Series][Number]
        # Example: DL01AB1234, KA02CD5678
        if len(text) >= 6:  # Minimum length for a valid plate
            # Check if first two characters are a valid state code
            potential_state_code = text[:2]
            
            if potential_state_code in self.indian_state_codes:
                # Try to format as XX00XX0000
                if len(text) >= 8:
                    # Extract numeric district code (typically 1-2 digits)
                    district_match = re.search(r'[A-Z]{2}(\d{1,2})', text)
                    if district_match:
                        district_start = 2
                        district_end = district_start + len(district_match.group(1))
                        
                        # Rest should be series (1-2 letters) and number (4 digits)
                        remaining = text[district_end:]
                        
                        # Try to separate series and number
                        series_match = re.search(r'([A-Z]{1,2})(\d{1,4})', remaining)
                        if series_match:
                            series = series_match.group(1)
                            number = series_match.group(2)
                            
                            # Format as XX-00-XX-0000
                            return f"{potential_state_code}-{district_match.group(1)}-{series}-{number}"
                
                # If specific formatting failed, try a general approach
                if len(text) >= 8:
                    # Format as XX-00-XX-0000 or similar based on length
                    if len(text) == 10:  # Standard length
                        return f"{text[:2]}-{text[2:4]}-{text[4:6]}-{text[6:]}"
                    elif len(text) == 9:  # Missing one character
                        return f"{text[:2]}-{text[2:4]}-{text[4:5]}-{text[5:]}"
                    elif len(text) == 8:  # Missing two characters
                        return f"{text[:2]}-{text[2:4]}-{text[4:]}"
        
        # If no pattern matched, return original text
        return text

    def process_frame(self, frame, draw_boxes=False):
        """Process a single frame and return detected plates without modifying the original image
        
        Args:
            frame: Input image frame
            draw_boxes: Whether to draw bounding boxes on the image (default: False)
            
        Returns:
            tuple: (processed_frame, detected_plates, plate_regions, confidences)
                - processed_frame: Original or annotated frame depending on draw_boxes
                - detected_plates: List of detected plate texts
                - plate_regions: List of plate coordinates (x1, y1, x2, y2)
                - confidences: List of confidence scores
        """
        # Create a copy of the frame if we need to draw on it
        if draw_boxes:
            processed_frame = frame.copy()
        else:
            processed_frame = frame
            
        # Detect plates
        plate_regions = self.detect_plates(frame)
        detected_plates = []
        confidences = []
        
        for plate_coords in plate_regions:
            x1, y1, x2, y2 = plate_coords
            
            # Extract text
            plate_text, confidence = self.extract_text(frame, plate_coords)
            
            if plate_text:
                detected_plates.append(plate_text)
                confidences.append(confidence)
                
                # Draw bounding box and text if requested
                if draw_boxes:
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add text above the bounding box
                    cv2.putText(processed_frame, plate_text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return processed_frame, detected_plates, plate_regions, confidences

    def save_results(self, filename, plate_number, confidence=0.0):
        """Save detection results to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            'filename': [filename],
            'plate_number': [plate_number],
            'confidence': [confidence],
            'timestamp': [timestamp]
        })
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        self.results_df.to_csv('output/detection_results.csv', index=False)

    def process_video(self, video_path, draw_boxes=False):
        """Process video file and return frames with annotations
        
        Args:
            video_path: Path to the video file
            draw_boxes: Whether to draw bounding boxes on the frames
            
        Returns:
            tuple: (frames, all_detected_plates)
                - frames: List of processed frames
                - all_detected_plates: List of detected plate texts for each frame
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        all_detected_plates = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, plates, _, confidences = self.process_frame(frame, draw_boxes)
            frames.append(processed_frame)
            all_detected_plates.append(plates)
            
            # Save results for each detected plate
            for i, plate in enumerate(plates):
                confidence = confidences[i] if i < len(confidences) else 0.0
                self.save_results(os.path.basename(video_path), plate, confidence)
        
        cap.release()
        return frames, all_detected_plates

    def process_image(self, image_input, draw_boxes=False):
        """Process single image and return annotated image and detected plates
        
        Args:
            image_input: Path to image file or image array
            draw_boxes: Whether to draw bounding boxes on the image
            
        Returns:
            tuple: (processed_frame, detected_plates, plate_regions, confidences)
                - processed_frame: Original or annotated frame
                - detected_plates: List of detected plate texts
                - plate_regions: List of plate coordinates
                - confidences: List of confidence scores for each plate
        """
        # Handle both file paths and direct image inputs
        if isinstance(image_input, str):
            # It's a file path
            frame = cv2.imread(image_input)
            filename = os.path.basename(image_input)
        else:
            # It's already an image array
            frame = image_input
            filename = "uploaded_image"
            
        processed_frame, plates, plate_regions, confidences = self.process_frame(frame, draw_boxes)
        
        # Save results for each detected plate
        for i, plate in enumerate(plates):
            confidence = confidences[i] if i < len(confidences) else 0.0
            self.save_results(filename, plate, confidence)
        
        return processed_frame, plates, plate_regions, confidences 