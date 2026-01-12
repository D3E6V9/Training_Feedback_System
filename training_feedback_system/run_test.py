import os
import sys
import cv2
import numpy as np
import django
from pathlib import Path

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'training_feedback_system.settings')
django.setup()

from feedback.rating_detector import AdvancedRatingDetector

def run_test():
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Read test image
    image_path = Path(__file__).parent / 'feedback' / 'test_images' / 'feedback_form.jpg'
    print(f"Reading image from: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image from {image_path}")
    
    # Run detection
    ratings, debug_info = detector.detect_ratings(img)
    
    # Print results
    print("\nRating Detection Results:")
    print("-" * 50)
    print("Q# | Rating | Confidence | Expected")
    print("-" * 50)
    
    expected = [5] * 8  # All ratings are 5 in this image
    correct = 0
    total_confidence = 0
    
    for i in range(8):
        rating = next((r for r in ratings if r['row'] == i), None)
        if rating:
            print(f"{i+1:2d} | {rating['rating']:^6d} | {rating['confidence']:^9.2f} | {expected[i]:^8d}")
            correct += 1 if rating['rating'] == expected[i] else 0
            total_confidence += rating['confidence']
    
    # Print summary
    accuracy = (correct / 8) * 100
    avg_confidence = total_confidence / len(ratings) if ratings else 0
    
    print(f"\nSummary:")
    print(f"Total Detected: {len(ratings)}/8")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    # Print debug info
    print("\nDetection Debug Info:")
    for key, value in debug_info.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"- {key}: {value}")

if __name__ == "__main__":
    run_test()
