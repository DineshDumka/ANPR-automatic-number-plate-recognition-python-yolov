import os
import requests
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def download_image(url, save_path):
    """Download an image from URL and save it to the specified path"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            print(f"Downloaded {save_path}")
            return True
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

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
        
    else:
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
    
    # Save the image
    img.save(output_path)
    print(f"Created synthetic plate: {output_path}")
    return True

def create_test_images():
    """Create test license plate images"""
    # Create directories if they don't exist
    os.makedirs("test_inputs", exist_ok=True)
    
    # List of sample license plate numbers
    plates = [
        "ABC123",
        "XYZ789",
        "DEF456",
        "GHI789",
        "JKL012"
    ]
    
    # Create synthetic license plates
    for i, plate in enumerate(plates):
        create_synthetic_license_plate(plate, f"test_inputs/plate_{i+1}.jpg", 
                                      plate_type="EU" if i % 2 == 0 else "US")
    
    # Download some real license plate images
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Cayman_Islands_license_plate_2.JPG/320px-Cayman_Islands_license_plate_2.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Delhi_registered_Hyundai_Santro_Xing_GL_Plus_in_Goa.jpg/320px-Delhi_registered_Hyundai_Santro_Xing_GL_Plus_in_Goa.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/2010-07-20_Black_Honda_Civic_Si_with_DC_plates.jpg/320px-2010-07-20_Black_Honda_Civic_Si_with_DC_plates.jpg"
    ]
    
    for i, url in enumerate(urls):
        download_image(url, f"test_inputs/real_plate_{i+1}.jpg")
    
    print("Test images created successfully!")

if __name__ == "__main__":
    create_test_images() 