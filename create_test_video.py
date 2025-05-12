import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_background(width, height):
    """Create a simple background for the video"""
    bg = np.ones((height, width, 3), dtype=np.uint8) * 100
    # Add some road-like texture
    for i in range(0, height, 20):
        cv2.line(bg, (0, i), (width, i), (80, 80, 80), 1)
    return bg

def create_license_plate(text, plate_type="EU"):
    """Create a license plate image with the given text"""
    if plate_type == "EU":
        # European style plate
        img = Image.new('RGB', (220, 70), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add blue band on left
        draw.rectangle([(0, 0), (30, 70)], fill=(0, 51, 153))
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((40, 15), text, fill=(0, 0, 0), font=font)
        
    else:  # US style
        # US style plate
        img = Image.new('RGB', (220, 70), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((35, 15), text, fill=(0, 0, 0), font=font)
    
    # Convert to numpy array for OpenCV
    return np.array(img)

def create_test_video():
    """Create a test video with moving license plates"""
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    
    # Create output directory if it doesn't exist
    os.makedirs("test_inputs", exist_ok=True)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_inputs/test_video.mp4', fourcc, fps, (width, height))
    
    # License plates to show in the video
    plates = [
        {"text": "ABC123", "type": "EU"},
        {"text": "XYZ789", "type": "US"}
    ]
    
    # Create license plate images
    plate_images = [create_license_plate(p["text"], p["type"]) for p in plates]
    
    # Initial positions
    positions = [
        {"x": 100, "y": 200, "dx": 2, "dy": 0},
        {"x": 400, "y": 300, "dx": -2, "dy": 0}
    ]
    
    # Generate video frames
    background = create_background(width, height)
    
    for frame_idx in range(fps * duration):
        # Create a copy of the background
        frame = background.copy()
        
        # Update positions and draw plates
        for i, (plate_img, pos) in enumerate(zip(plate_images, positions)):
            # Update position
            pos["x"] += pos["dx"]
            pos["y"] += pos["dy"]
            
            # Bounce off edges
            if pos["x"] <= 0 or pos["x"] + plate_img.shape[1] >= width:
                pos["dx"] *= -1
            if pos["y"] <= 0 or pos["y"] + plate_img.shape[0] >= height:
                pos["dy"] *= -1
            
            # Draw the plate on the frame
            h, w = plate_img.shape[:2]
            frame[pos["y"]:pos["y"]+h, pos["x"]:pos["x"]+w] = plate_img
        
        # Write the frame
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Created test video: test_inputs/test_video.mp4")

if __name__ == "__main__":
    create_test_video() 