import cv2
import os
import matplotlib.pyplot as plt
from anpr import ANPR

def test_anpr_with_image(image_path):
    """Test the ANPR system with a sample image"""
    # Initialize ANPR system
    anpr_system = ANPR()
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Process the image
    processed_image, plates, plate_regions = anpr_system.process_image(image, draw_boxes=True)
    
    # Print results
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Detected plates: {plates}")
    
    # Display the image with detected plates
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected plates: {', '.join(plates) if plates else 'None'}")
    plt.axis('off')
    
    # Save the result
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, processed_image)
    print(f"Result saved to {output_path}")
    
    return plates, plate_regions

if __name__ == "__main__":
    # Test with all images in test_inputs directory
    test_dir = "test_inputs"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        exit(1)
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Error: No image files found in {test_dir}")
        exit(1)
    
    print(f"Found {len(image_files)} image files in {test_dir}")
    
    # Test with the first image
    test_image = os.path.join(test_dir, image_files[0])
    plates, regions = test_anpr_with_image(test_image)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Image: {os.path.basename(test_image)}")
    print(f"Detected plates: {plates}")
    print(f"Number of plate regions: {len(regions)}")
    
    print("\nTest completed successfully!") 