import cv2
import numpy as np
from rating_detector import AdvancedRatingDetector
import os
from pathlib import Path

def test_ratings():
    # Load the image
    image_path = str(Path(__file__).parent / 'test_images' / 'feedback_form.jpg')
    print(f"Reading feedback form from: {image_path}")
    
    # Create a fresh test image to simulate the feedback form
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw grid lines
    cell_height = img.shape[0] // 8  # 8 rows
    cell_width = img.shape[1] // 5   # 5 columns (for ratings 1-5)
    
    # Draw horizontal lines
    for i in range(9):
        y = i * cell_height
        cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 0), 1)
    
    # Draw vertical lines
    for i in range(6):
        x = i * cell_width
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 0), 1)
    
    # Add checkmarks in column 5 (strongly agree) for all rows
    for row in range(8):
        x = cell_width // 2
        y = row * cell_height + cell_height // 2
          # Draw a more prominent checkmark
        size = 15
        pt1 = (x - size, y)
        pt2 = (x, y + size)
        pt3 = (x + size * 2, y - size)
        cv2.line(img, pt1, pt2, (0, 0, 0), 3)
        cv2.line(img, pt2, pt3, (0, 0, 0), 3)
        
        # Add some noise around the checkmark to make it more realistic
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    
    # Save the test image
    cv2.imwrite(image_path, img)
    print("Created test image with all ratings marked as 5")
    
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Run detection
    print("\nRunning rating detection...")
    ratings, debug_info = detector.detect_ratings(img)
    
    # Print results
    print("\nDetected Ratings:")
    print("-" * 50)
    print("Question | Rating | Confidence")
    print("-" * 50)
    
    for row in range(8):
        rating = next((r for r in ratings if r['row'] == row), None)
        if rating:
            print(f"{row + 1:8d} | {rating['rating']:6d} | {rating['confidence']:9.2f}")
        else:
            print(f"{row + 1:8d} | {'N/A':6s} | {'N/A':9s}")
    
    # Save visualization
    viz_img = detector.visualize_detections(img, ratings)
    viz_path = str(Path(__file__).parent / 'test_images' / 'feedback_form_detected.jpg')
    cv2.imwrite(viz_path, viz_img)
    print(f"\nVisualization saved to: {viz_path}")
    
    return ratings, debug_info

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--actual":
        # Load and process the actual feedback form
        actual_path = str(Path(__file__).parent / 'test_images' / 'actual_form.jpg')
        img = cv2.imread(actual_path)
        if img is not None:
            print(f"\nTesting with actual feedback form: {actual_path}")
            detector = AdvancedRatingDetector()
            ratings, debug_info = detector.detect_ratings(img)
            
            print("\nDetected Ratings (Actual Form):")
            print("-" * 50)
            print("Question | Rating | Confidence")
            print("-" * 50)
            
            for row in range(8):
                rating = next((r for r in ratings if r['row'] == row), None)
                if rating:
                    print(f"{row + 1:8d} | {rating['rating']:6d} | {rating['confidence']:9.2f}")
                else:
                    print(f"{row + 1:8d} | {'N/A':6s} | {'N/A':9s}")
            
            # Save visualization
            viz_img = detector.visualize_detections(img, ratings)
            viz_path = str(Path(__file__).parent / 'test_images' / 'feedback_form_test_detected.jpg')
            cv2.imwrite(viz_path, viz_img)
            print(f"\nVisualization saved to: {viz_path}")
        else:
            print(f"Error: Could not read actual feedback form from {actual_path}")
    else:
        # Run with synthetic test image
        test_ratings()
