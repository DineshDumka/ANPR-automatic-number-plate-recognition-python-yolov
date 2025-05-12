import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def create_synthetic_license_plate(text, output_path, plate_type="EU"):
    """Create a synthetic license plate image with the given text"""
    if plate_type == "EU":
        # European style plate
        img = Image.new('RGB', (440, 140), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add blue band on left
        draw.rectangle([(0, 0), (60, 140)], fill=(0, 51, 153))
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 80)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((80, 30), text, fill=(0, 0, 0), font=font)
        
    elif plate_type == "US":
        # US style plate
        img = Image.new('RGB', (440, 140), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add colored background
        draw.rectangle([(0, 0), (440, 140)], fill=(220, 220, 220))
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 80)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((70, 30), text, fill=(0, 0, 0), font=font)
    
    else:  # Asian style
        # Asian style plate
        img = Image.new('RGB', (440, 140), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add colored background
        draw.rectangle([(0, 0), (440, 140)], fill=(255, 255, 200))
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 70)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((50, 40), text, fill=(0, 0, 0), font=font)
    
    # Save the image
    img.save(output_path)
    
    # Add some noise and blur to make it more realistic
    cv_img = cv2.imread(output_path)
    # Add slight blur
    cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)
    # Add some noise
    noise = np.random.normal(0, 5, cv_img.shape).astype(np.uint8)
    cv_img = cv2.add(cv_img, noise)
    # Save the processed image
    cv2.imwrite(output_path, cv_img)
    
    print(f"Created synthetic plate: {output_path}")
    return True

def create_test_plates():
    """Create test license plate images"""
    # Create directory if it doesn't exist
    os.makedirs("test_inputs", exist_ok=True)
    
    # List of sample license plate numbers with different formats
    plates = [
        {"text": "ABC123", "type": "EU"},
        {"text": "XYZ789", "type": "US"},
        {"text": "DEF456", "type": "EU"},
        {"text": "GHI789", "type": "US"},
        {"text": "JKL012", "type": "EU"},
        {"text": "MNO345", "type": "US"},
        {"text": "PQR678", "type": "EU"},
        {"text": "STU901", "type": "US"}
    ]
    
    # Create synthetic license plates
    for i, plate in enumerate(plates):
        create_synthetic_license_plate(plate["text"], f"test_inputs/plate_{i+1}.jpg", plate["type"])
    
    print(f"Created {len(plates)} test license plates in the test_inputs directory.")

if __name__ == "__main__":
    create_test_plates() 