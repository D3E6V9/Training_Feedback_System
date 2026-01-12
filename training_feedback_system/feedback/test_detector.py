import cv2
import numpy as np
from rating_detector import AdvancedRatingDetector

def test_and_visualize_detection(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image from {image_path}")
        return
    
    # Create a copy for visualization
    viz_img = img.copy()
    
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Get ratings and debug info
    ratings, debug_info = detector.detect_ratings(img)
    
    # Draw detected marks
    height, width = img.shape[:2]
    cell_height = height // 8
    cell_width = width // 6
    
    # Define colors for visualization
    COLORS = {
        5: (0, 255, 0),    # Green for correct detection
        4: (255, 165, 0),  # Orange for wrong detection
        0: (0, 0, 255)     # Red for missed detection
    }
    
    # Expected ratings for this form
    expected_ratings = [5] * 8  # All ratings are 5 in this image
    
    print("\nRating Detection Results:")
    print("------------------------")
    print("Row | Detected | Expected | Confidence")
    print("----|---------|-----------|-----------")
    
    for row in range(8):
        # Find rating for this row
        row_rating = next((r for r in ratings if r['row'] == row), {'rating': 0, 'confidence': 0})
        detected = row_rating['rating']
        confidence = row_rating['confidence']
        expected = expected_ratings[row]
        
        # Print results
        print(f" {row+1}  |    {detected}    |     {expected}     |   {confidence:.2f}")
        
        # Draw rectangle around detected mark
        if detected > 0:
            col = 6 - detected  # Convert rating (5-1) to column index (1-5)
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            
            # Color based on correctness
            color = COLORS[5] if detected == expected else COLORS[4]
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
    
    # Calculate accuracy
    correct_detections = sum(1 for r in ratings if r['rating'] == expected_ratings[r['row']])
    accuracy = (correct_detections / 8) * 100
    
    print("\nDetection Statistics:")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average Confidence: {np.mean([r['confidence'] for r in ratings]):.2f}")
    
    # Save visualization
    output_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, viz_img)
    print(f"\nVisualization saved to: {output_path}")
    
if __name__ == "__main__":
    # Use the test image path
    image_path = "test_images/feedback_form.jpg"
    test_and_visualize_detection(image_path)
