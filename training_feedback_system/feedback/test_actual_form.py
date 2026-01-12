import cv2
import numpy as np
from rating_detector import AdvancedRatingDetector
import os
from pathlib import Path

def test_actual_form():
    """Test rating detection on the actual feedback form"""
    
    # Get the image path
    image_path = Path(__file__).parent / 'test_images' / 'feedback_form.jpg'
    print(f"Testing image: {image_path}")
    
    # Try to read the file content
    try:
        with open(image_path, 'rb') as f:
            content = f.read()
            print(f"\nFile content starts with: {content[:50]}")
            print(f"Content length: {len(content)}")
            if content.startswith(b'data:image'):
                print("File appears to contain base64 data instead of binary image data")
                return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        print(f"File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'}")
        return
        
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Run detection
    print("\nRunning detection...")
    ratings, debug_info = detector.detect_ratings(img)
    
    # Print results
    print("\nDetected Ratings:")
    for rating in sorted(ratings, key=lambda x: x['row']):
        print(f"Row {rating['row'] + 1}: Rating {rating['rating']} (Confidence: {rating['confidence']:.2f})")
    
    # Save visualization
    viz_img = detector.visualize_detections(img, ratings)
    output_path = Path(__file__).parent / 'test_images' / 'feedback_form_detected.jpg'
    cv2.imwrite(str(output_path), viz_img)
    print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    test_actual_form()
