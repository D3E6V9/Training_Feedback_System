import os
import cv2
import numpy as np
from pathlib import Path
from rating_detector import AdvancedRatingDetector

def test_feedback_form():
    """Test rating detection on actual feedback form"""
    
    # Get path to test image
    test_dir = Path(__file__).parent / 'test_images'
    image_path = test_dir / 'feedback_form.jpg'
    
    print(f"Testing with image: {image_path}")
    
    # Read test image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Run detection
    print("\nRunning detection...")
    ratings, debug_info = detector.detect_ratings(img)
    
    # Expected ratings for this form
    expected = [5] * 8  # All marks are in column 1 (rating 5)
    
    # Print results
    print("\nRating Detection Results:")
    print("-" * 60)
    print("Q# | Rating | Expected | Confidence | Match")
    print("-" * 60)
    
    total_confidence = 0
    correct = 0
    
    for row in range(8):
        rating = next((r for r in ratings if r['row'] == row), None)
        detected = rating['rating'] if rating else 0
        confidence = rating['confidence'] if rating else 0.0
        match = "✓" if detected == expected[row] else "✗"
        
        print(f"{row+1:2d} | {detected:^6d} | {expected[row]:^8d} | {confidence:^9.2f} | {match}")
        
        if detected == expected[row]:
            correct += 1
        total_confidence += confidence if rating else 0
    
    # Print summary
    accuracy = (correct / 8) * 100
    avg_confidence = total_confidence / len(ratings) if ratings else 0
    
    print("\nTest Summary:")
    print(f"Total Marks Detected: {len(ratings)}/8")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    # Generate visualization
    viz_img = detector.visualize_detections(img, ratings)
    output_path = test_dir / 'feedback_form_detected.jpg'
    cv2.imwrite(str(output_path), viz_img)
    print(f"\nVisualization saved to: {output_path}")
    
    return ratings, debug_info

if __name__ == "__main__":
    test_feedback_form()
