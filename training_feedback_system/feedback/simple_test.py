import os
import cv2
import numpy as np
import django
from pathlib import Path

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'training_feedback_system.settings')
django.setup()

from rating_detector import AdvancedRatingDetector

def test_feedback_detection(image_path):
    """Test rating detection on the feedback form"""
    print(f"Reading image from: {image_path}")
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
        
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}")
    
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Detect ratings
    print("\nRunning detection...")
    ratings, debug_info = detector.detect_ratings(img)
    
    # Expected ratings (all 5's in this form)
    expected = [5] * 8
    
    # Print results
    print("\nRating Detection Results:")
    print("-" * 60)
    print("Question | Detected | Expected | Confidence | Match")
    print("-" * 60)
    
    correct = 0
    total_confidence = 0
    
    for row in range(8):
        rating = next((r for r in ratings if r['row'] == row), None)
        if rating:
            detected = rating['rating']
            confidence = rating['confidence']
            match = "✓" if detected == expected[row] else "✗"
            if detected == expected[row]:
                correct += 1
            total_confidence += confidence
        else:
            detected = 0
            confidence = 0.0
            match = "✗"
            
        print(f"   {row+1:2d}   |    {detected}    |    {expected[row]}    |   {confidence:.2f}   | {match}")
    
    accuracy = (correct / 8) * 100
    avg_confidence = total_confidence / len(ratings) if ratings else 0
    
    print("\nSummary:")
    print(f"Total marks detected: {len(ratings)}/8")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average confidence: {avg_confidence:.2f}")
    
    # Save visualization
    viz_img = detector.visualize_detections(img, ratings)
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, viz_img)
    print(f"\nVisualization saved to: {output_path}")
    
    return ratings, debug_info

if __name__ == "__main__":
    image_path = r"c:\Users\HP\Desktop\Resolution 2025\Feedbacks\training_feedback_system\feedback\test_images\feedback_form_test.jpg"
    test_feedback_detection(image_path)
