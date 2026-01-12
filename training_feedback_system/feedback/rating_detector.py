import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class DetectionDebug:
    preprocessing_info: Dict[str, Any]
    detection_steps: List[str]
    confidence_scores: List[float]
    mark_positions: List[Tuple[int, int]]

class AdvancedRatingDetector:
    def __init__(self):
        self.debug_info = {
            'preprocessing': {},
            'detection_steps': [],
            'confidence_scores': [],
            'mark_positions': []
        }
        # Configure weights for different detection features
        self.weights = {
            'checkmark': 0.4,    # Weight for checkmark pattern detection
            'darkness': 0.3,     # Weight for dark pixel density
            'center': 0.15,      # Weight for mark being near cell center
            'edge': 0.15         # Weight for edge detection
        }
        self.detection_threshold = 0.15  # Minimum confidence threshold for initial detection

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image preprocessing pipeline optimized for checkmark detection.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Use Gaussian adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        self.debug_info['preprocessing'].update({
            'original_size': image.shape,
            'enhancement': 'CLAHE(3.0)',
            'denoising': 'fastNlMeans',
            'thresholding': 'adaptive_gaussian'
        })
        
        return thresh

    def detect_ratings(self, image: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Detect ratings in the feedback form image.
        Returns list of detected ratings and debug info.
        """
        # Reset debug info
        self.debug_info = {
            'preprocessing': {},
            'detection_steps': [],
            'confidence_scores': [],
            'mark_positions': []
        }
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Get image dimensions
        height, width = preprocessed.shape
        cell_height = height // 8  # 8 rows
        cell_width = width // 5    # 5 columns
        
        ratings = []
        for row in range(8):
            max_confidence = 0
            detected_rating = 0
            detected_col = 0
            
            for col in range(5):
                # Extract cell region
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height
                cell = preprocessed[y1:y2, x1:x2]
                
                # Calculate features
                checkmark_score = self._detect_checkmark(cell)
                darkness_score = np.sum(cell > 0) / (cell_width * cell_height)
                center_score = self._calculate_center_score(cell)
                edge_score = self._calculate_edge_score(cell)
                
                # Calculate weighted confidence
                confidence = (
                    self.weights['checkmark'] * checkmark_score +
                    self.weights['darkness'] * darkness_score +
                    self.weights['center'] * center_score +
                    self.weights['edge'] * edge_score
                )
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_rating = 5 - col  # Convert column to rating (rightmost is 1)
                    detected_col = col
            
            if max_confidence >= self.detection_threshold:
                ratings.append({
                    'row': row,
                    'rating': detected_rating,
                    'confidence': max_confidence,
                    'column': detected_col
                })
                self.debug_info['mark_positions'].append((detected_col, row))
                self.debug_info['confidence_scores'].append(max_confidence)
        
        return ratings, self.debug_info

    def _detect_checkmark(self, cell: np.ndarray) -> float:
        """
        Detect checkmark pattern in a cell.
        Returns confidence score between 0 and 1.
        """
        # Find contours
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # Get the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find diagonal lines using Hough transform
        edges = cv2.Canny(cell, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is None:
            return 0.0
            
        # Count diagonal lines (approximately 45 degrees)
        diagonal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 30 <= angle <= 60 or 120 <= angle <= 150:
                diagonal_count += 1
        
        # Calculate score based on diagonal lines and contour properties
        score = min(1.0, (diagonal_count / 2) * 0.7 + (cv2.contourArea(max_contour) / cell.size) * 0.3)
        return score

    def _calculate_center_score(self, cell: np.ndarray) -> float:
        """
        Calculate how close the mark is to the cell center.
        """
        height, width = cell.shape
        center_y, center_x = height // 2, width // 2
        
        # Create a distance matrix from the center
        y, x = np.ogrid[:height, :width]
        dist_matrix = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Weight pixels by their distance from center
        weighted = np.sum((1 - dist_matrix/max_dist) * (cell > 0))
        return min(1.0, weighted / (width * height))

    def _calculate_edge_score(self, cell: np.ndarray) -> float:
        """
        Calculate edge detection score.
        """
        edges = cv2.Canny(cell, 50, 150)
        return min(1.0, np.sum(edges > 0) / (cell.shape[0] * cell.shape[1] * 0.2))

    def visualize_detections(self, image: np.ndarray, ratings: List[Dict]) -> np.ndarray:
        """
        Draw detection results on the image.
        """
        viz_img = image.copy()
        height, width = image.shape[:2]
        cell_height = height // 8
        cell_width = width // 5
        
        # Draw grid
        for i in range(9):
            y = i * cell_height
            cv2.line(viz_img, (0, y), (width, y), (0, 0, 255), 1)
        for i in range(6):
            x = i * cell_width
            cv2.line(viz_img, (x, 0), (x, height), (0, 0, 255), 1)
        
        # Draw detected marks
        for rating in ratings:
            row = rating['row']
            col = 5 - rating['rating']  # Convert rating back to column
            confidence = rating['confidence']
            
            # Calculate cell center
            cx = int((col + 0.5) * cell_width)
            cy = int((row + 0.5) * cell_height)
            
            # Color based on confidence
            if confidence >= 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw circle and confidence score
            cv2.circle(viz_img, (cx, cy), 15, color, 2)
            cv2.putText(viz_img, f"{confidence:.2f}", (cx-20, cy+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return viz_img
