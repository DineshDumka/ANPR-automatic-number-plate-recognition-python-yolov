"""
Indian License Plate Detection Model

This script provides functionality to fine-tune a YOLOv5/YOLOv8 model 
specifically for Indian license plate detection.

Usage:
    python indian_plate_model.py --train  # Train the model
    python indian_plate_model.py --test   # Test the model
"""

import os
import argparse
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path
import cv2

def setup_model_directories():
    """Create necessary directories for model training and data"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/images/train', exist_ok=True)
    os.makedirs('data/images/val', exist_ok=True)
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)
    
    # Create data.yaml file for training
    data_yaml = {
        'train': './data/images/train',
        'val': './data/images/val',
        'nc': 1,  # number of classes (just license plate)
        'names': ['license_plate']
    }
    
    with open('data/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("Model directories set up successfully.")

def download_pretrained_model():
    """Download pretrained YOLOv5 or YOLOv8 model"""
    try:
        # Try YOLOv5n
        model = YOLO('yolov5nu.pt')
        print("Downloaded YOLOv5nu model successfully.")
        return model
    except:
        try:
            # Fall back to YOLOv8n
            model = YOLO('yolov8n.pt')
            print("Downloaded YOLOv8n model successfully.")
            return model
        except Exception as e:
            print(f"Error downloading pretrained model: {e}")
            return None

def train_model(epochs=50, batch_size=16, img_size=640):
    """Train the model on Indian license plate data"""
    # Check if we have any training data
    train_dir = Path('data/images/train')
    if not train_dir.exists() or len(list(train_dir.glob('*'))) == 0:
        print("No training data found. Please add images to data/images/train and labels to data/labels/train.")
        return
    
    # Download pretrained model
    model = download_pretrained_model()
    if model is None:
        return
    
    # Define training parameters
    model.train(
        data='data/data.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,  # Early stopping patience
        save=True,  # Save best model
        project='models',
        name='indian_plates'
    )
    
    print("Model training completed.")
    
def test_model(img_path=None):
    """Test the model on a sample image or test dataset"""
    # Check if we have a trained model
    model_path = Path('models/indian_plates/weights/best.pt')
    if not model_path.exists():
        print("No trained model found. Please train the model first.")
        # Fall back to pretrained model
        model = download_pretrained_model()
    else:
        # Load fine-tuned model
        model = YOLO(model_path)
        print("Loaded fine-tuned Indian license plate model.")
    
    if model is None:
        return
    
    # Test on a single image if provided
    if img_path and os.path.exists(img_path):
        results = model.predict(img_path, conf=0.25)
        print(f"Detection results for {img_path}:")
        for result in results:
            print(f"Found {len(result.boxes)} license plates.")
            
            # Save the result image
            result_img = result.plot()
            save_path = os.path.join('output', f"detected_{os.path.basename(img_path)}")
            os.makedirs('output', exist_ok=True)
            cv2.imwrite(save_path, result_img)
            print(f"Result saved to {save_path}")
    else:
        # Test on validation set
        val_dir = Path('data/images/val')
        if val_dir.exists() and len(list(val_dir.glob('*'))) > 0:
            results = model.val()
            print("Validation results:")
            print(f"mAP@0.5: {results.box.map50}")
            print(f"mAP@0.5:0.95: {results.box.map}")
        else:
            print("No validation data or test image found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indian License Plate Detection Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--img', type=str, help='Path to test image')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_model_directories()
    
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch)
    
    if args.test:
        test_model(img_path=args.img)
    
    if not args.train and not args.test:
        print("No action specified. Use --train or --test.")
        print("Example usage:")
        print("  python indian_plate_model.py --train")
        print("  python indian_plate_model.py --test --img test_inputs/plate_1.jpg") 