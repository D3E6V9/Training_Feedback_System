import cv2
import numpy as np
from rating_detector import AdvancedRatingDetector

def main():
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Create a test image (all white background)
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Draw a sample table grid
    cell_height = img.shape[0] // 8
    cell_width = img.shape[1] // 6
    
    # Draw horizontal lines
    for i in range(9):
        y = i * cell_height
        cv2.line(img, (0, y), (img.shape[1], y), (0, 0, 0), 1)
    
    # Draw vertical lines
    for i in range(7):
        x = i * cell_width
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 0), 1)
    
    # Add checkmarks (simulating the actual form)
    for row in range(8):
        # Add checkmark in first column (rating 5)
        x = cell_width // 2
        y = row * cell_height + cell_height // 2
        
        # Draw checkmark
        cv2.line(img, (x-10, y), (x, y+10), (0, 0, 0), 2)
        cv2.line(img, (x, y+10), (x+20, y-10), (0, 0, 0), 2)
    
    # Run detection
    ratings, debug_info = detector.detect_ratings(img)
    
    # Print results
    print("\nRating Detection Results:")
    print("-" * 50)
    print("Q# | Rating | Confidence")
    print("-" * 50)
    
    for i in range(8):
        rating = next((r for r in ratings if r['row'] == i), None)
        if rating:
            print(f"{i+1:2d} | {rating['rating']:^6d} | {rating['confidence']:^9.2f}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total Detected: {len(ratings)}/8")
    avg_confidence = np.mean([r['confidence'] for r in ratings]) if ratings else 0
    print(f"Average Confidence: {avg_confidence:.2f}")
    
    # Save visualization
    viz_img = detector.visualize_detections(img, ratings)
    cv2.imwrite("test_output.jpg", viz_img)
    print("\nVisualization saved as: test_output.jpg")

if __name__ == "__main__":
    main()
