import cv2
import numpy as np
from PIL import Image
import io
import os
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of Indian state codes
INDIAN_STATE_CODES = [
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 
    'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 
    'TS', 'UK', 'UP', 'WB'
]

def create_output_dir():
    """Create output directory if it doesn't exist"""
    try:
        os.makedirs('output', exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating output directory: {str(e)}")
        return False

def save_processed_image(image, filename):
    """Save processed image to output directory"""
    try:
        output_path = os.path.join('output', filename)
        cv2.imwrite(output_path, image)
        return output_path
    except Exception as e:
        logger.error(f"Error saving image {filename}: {str(e)}")
        return None

def save_processed_video(frames, filename, fps=30):
    """Save processed video frames to output directory"""
    if not frames:
        logger.warning("No frames provided to save_processed_video")
        return None
        
    try:
        output_path = os.path.join('output', filename)
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return output_path
    except Exception as e:
        logger.error(f"Error saving video {filename}: {str(e)}")
        return None

def convert_to_bytes(image):
    """Convert OpenCV image to bytes for Streamlit display"""
    try:
        is_success, buffer = cv2.imencode(".jpg", image)
        if is_success:
            return buffer.tobytes()
        return None
    except Exception as e:
        logger.error(f"Error converting image to bytes: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return appropriate data"""
    if uploaded_file is None:
        return None
    
    try:
        file_bytes = uploaded_file.getvalue()
        
        if uploaded_file.type.startswith('image/'):
            # Process image
            image = Image.open(io.BytesIO(file_bytes))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif uploaded_file.type.startswith('video/'):
            # Save video temporarily
            temp_path = os.path.join('output', 'temp_video.mp4')
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            return temp_path
        
        return None
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return None

def get_file_extension(filename):
    """Get file extension from filename"""
    try:
        return os.path.splitext(filename)[1].lower()
    except Exception as e:
        logger.error(f"Error getting file extension: {str(e)}")
        return ""

def is_valid_image(filename):
    """Check if file is a valid image"""
    return get_file_extension(filename) in ['.jpg', '.jpeg', '.png']

def is_valid_video(filename):
    """Check if file is a valid video"""
    return get_file_extension(filename) in ['.mp4', '.avi', '.mov']

def extract_frames_from_video(video_path, max_frames=100):
    """Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        list: Extracted frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval to extract max_frames evenly distributed frames
        interval = max(1, total_frames // max_frames)
        
        frames = []
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % interval == 0:
                frames.append(frame)
                
            count += 1
            if len(frames) >= max_frames:
                break
                
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames from video {video_path}: {str(e)}")
        return []

def is_indian_license_plate(text):
    """Check if the text matches Indian license plate pattern"""
    # Remove all non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum())
    text = text.upper()
    
    # Check minimum length
    if len(text) < 6:
        return False
    
    # Check if first two characters are a valid state code
    state_code = text[:2]
    if state_code not in INDIAN_STATE_CODES:
        return False
    
    # Check if next 1-2 characters are digits (district code)
    district_match = re.match(r'^[A-Z]{2}(\d{1,2})', text)
    if not district_match:
        return False
    
    # Rest should have at least one letter and some digits
    remaining = text[2 + len(district_match.group(1)):]
    if not re.search(r'[A-Z]', remaining) or not re.search(r'\d', remaining):
        return False
    
    return True

def enhance_plate_image(plate_img):
    """Apply multiple preprocessing techniques to enhance license plate image for OCR
    
    Args:
        plate_img: License plate image
        
    Returns:
        list: List of preprocessed images
    """
    enhanced_images = []
    
    # Original image
    enhanced_images.append(plate_img)
    
    # Grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    enhanced_images.append(gray)
    
    # Resize to larger dimensions for better OCR
    h, w = plate_img.shape[:2]
    resized = cv2.resize(plate_img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    enhanced_images.append(resized)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    enhanced_images.append(bilateral)
    
    # Apply histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)
    enhanced_images.append(equalized)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    enhanced_images.append(thresh)
    
    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_images.append(otsu)
    
    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    enhanced_images.append(morph_close)
    
    # Apply edge enhancement
    edges = cv2.Canny(gray, 100, 200)
    enhanced_images.append(edges)
    
    return enhanced_images

def format_indian_plate(text):
    """Format text according to Indian license plate pattern
    
    Args:
        text: Detected license plate text
        
    Returns:
        str: Formatted license plate text
    """
    # Remove all non-alphanumeric characters
    text = ''.join(c for c in text if c.isalnum())
    text = text.upper()
    
    # Check if it's a valid Indian plate format
    if not is_indian_license_plate(text):
        return text
    
    # Extract state code (first 2 characters)
    state_code = text[:2]
    
    # Extract district code (1-2 digits after state code)
    district_match = re.match(r'^[A-Z]{2}(\d{1,2})', text)
    if not district_match:
        return text
    
    district_code = district_match.group(1)
    remaining = text[2 + len(district_code):]
    
    # Try to separate series and number
    series_match = re.search(r'([A-Z]{1,2})(\d{1,4})', remaining)
    if series_match:
        series = series_match.group(1)
        number = series_match.group(2)
        return f"{state_code}-{district_code}-{series}-{number}"
    
    # If specific formatting failed, try a general approach
    if len(text) == 10:  # Standard length
        return f"{text[:2]}-{text[2:4]}-{text[4:6]}-{text[6:]}"
    elif len(text) == 9:  # Missing one character
        return f"{text[:2]}-{text[2:4]}-{text[4:5]}-{text[5:]}"
    elif len(text) == 8:  # Missing two characters
        return f"{text[:2]}-{text[2:4]}-{text[4:]}"
    
    return text 