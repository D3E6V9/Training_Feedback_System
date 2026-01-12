import os
import sys
import cv2
import numpy as np

# Add parent directory to Python path to find the rating_detector module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from feedback.rating_detector import AdvancedRatingDetector

def test_rating_detection(image_path):
    """Test rating detection on a single image and return visualization."""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Initialize detector
    detector = AdvancedRatingDetector()
    
    # Get ratings and debug info
    ratings, debug_info = detector.detect_ratings(img)
    
    # Generate visualization
    viz_img = detector.visualize_detections(img, ratings)
    
    # Save visualization
    base_name = os.path.basename(image_path)
    viz_path = os.path.join('test_images', f'detected_{base_name}')
    cv2.imwrite(viz_path, viz_img)
    
    # Expected ratings for the test form
    expected_ratings = [5] * 8  # All ratings are 5 in this image
    
    print("\nRating Detection Results:")
    print("------------------------")
    print("Row | Detected | Expected | Confidence")
    print("----|---------|-----------|-----------")
    
    correct_count = 0
    for row in range(8):
        rating = next((r for r in ratings if r['row'] == row), None)
        detected = rating['rating'] if rating else 0
        confidence = rating['confidence'] if rating else 0.0
        expected = expected_ratings[row]
        
        if detected == expected:
            correct_count += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f" {row+1}  |    {detected}    |     {expected}     |   {confidence:.2f}  {status}")
    
    accuracy = (correct_count / 8) * 100
    
    # Print summary
    confidence_scores = [r['confidence'] for r in ratings]
    if confidence_scores:
        print("\nDetection Summary:")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Total marks detected: {len(ratings)}")
        print(f"Average confidence: {np.mean(confidence_scores):.2f}")
        print(f"Min confidence: {min(confidence_scores):.2f}")
        print(f"Max confidence: {max(confidence_scores):.2f}")
    
    print(f"\nVisualization saved to: {viz_path}")
    return ratings, debug_info, viz_path

if __name__ == "__main__":
    # Test with the feedback form image using absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(current_dir, 'test_images', 'feedback_form.jpg')
    try:
        ratings, debug_info, viz_path = test_rating_detection(test_image)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()