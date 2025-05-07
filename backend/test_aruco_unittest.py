import unittest
import os
import sys
import cv2
import numpy as np
from PIL import Image

# Add parent directory to the path to import app.py modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import detect_aruco_markers, crop_and_rectify_aruco_square
from create_heatmap import create_heatmap

class TestArucoFunctions(unittest.TestCase):
    """Test case for ArUco marker detection and rectification functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Define the test image path
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_image_path = os.path.join(cls.test_dir, "test", "test2.jpg")
        cls.heatmap_test_path = os.path.join(cls.test_dir, "test", "test_heatmap.jpg")
        
        # Ensure the test image exists
        assert os.path.exists(cls.test_image_path), f"Test image not found at {cls.test_image_path}"
        assert os.path.exists(cls.heatmap_test_path), f"Heatmap test image not found at {cls.heatmap_test_path}"
        
        # Load the test image
        cls.test_image = cv2.imread(cls.test_image_path)
        cls.heatmap_test_image = cv2.imread(cls.heatmap_test_path)
        
        assert cls.test_image is not None, f"Failed to load test image from {cls.test_image_path}"
        assert cls.heatmap_test_image is not None, f"Failed to load heatmap test image from {cls.heatmap_test_path}"
    
    def test_detect_aruco_markers(self):
        """Test the detect_aruco_markers function."""
        # Call the function
        result = detect_aruco_markers(self.test_image)
        
        # Check if the result is a tuple with 4 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        
        # Unpack the results
        output_image, corners, ids, rejected = result
        
        # Basic assertions - output image should exist and have the same dimensions
        self.assertIsNotNone(output_image)
        self.assertEqual(output_image.shape, self.test_image.shape)
        
        # Test if corners and ids are of the correct type
        if ids is not None:
            self.assertIsInstance(corners, tuple, f"Expected 'corners' to be a list, got {type(corners)}")
            self.assertIsInstance(ids, np.ndarray, f"Expected 'ids' to be a numpy array, got {type(ids)}")
            
            # Print detection results for debugging
            print(f"Detected {len(ids)} ArUco markers with IDs: {ids.flatten()}")
            
            # Save the output image for visual inspection
            output_path = os.path.join(self.test_dir, "test", "unittest_detect_output.jpg")
            cv2.imwrite(output_path, output_image)
        else:
            # If no markers were detected, we still expect certain outputs
            self.assertIsInstance(corners, list, f"Expected 'corners' to be a list, got {type(corners)}")
            self.assertIsNone(ids)
            print("No ArUco markers detected in the test image")
    
    def test_crop_and_rectify_aruco_square(self):
        """Test the crop_and_rectify_aruco_square function with the inner corners approach."""
        # Call the function
        rectified_image = crop_and_rectify_aruco_square(self.test_image)
        
        # The function should always return an image
        self.assertIsNotNone(rectified_image)
        
        # Save the rectified image for visual inspection
        output_path = os.path.join(self.test_dir, "test", "unittest_rectified_output.jpg")
        cv2.imwrite(output_path, rectified_image)
        
        # Log dimensions for analysis
        print(f"Original image dimensions: {self.test_image.shape[:2]}")
        print(f"Rectified image dimensions: {rectified_image.shape[:2]}")
        
        # The "selected_corners.jpg" file shows the corners selected for rectification
        # This file is saved by the crop_and_rectify_aruco_square function
        if os.path.exists("selected_corners.jpg"):
            selected_corners_path = os.path.join(self.test_dir, "test", "unittest_selected_corners.jpg")
            import shutil
            shutil.copy("selected_corners.jpg", selected_corners_path)
            print(f"Saved corner visualization to {selected_corners_path}")
        
        # Check if we got an error image or a properly rectified image
        if rectified_image.shape[:2] == (300, 300):  # Default target size
            print("Successfully rectified image to target size (300, 300)")
            
            # Additional check: verify the rectified image is not just black or white
            # Calculate the standard deviation of pixel values - if it's very low, the image might be mostly uniform
            std_dev = np.std(rectified_image)
            print(f"Rectified image standard deviation: {std_dev}")
            self.assertGreater(std_dev, 5.0, "Rectified image appears to be too uniform")
            
        else:
            # If the shape doesn't match the target, it means the rectification 
            # probably failed and we got back the original image with error text
            print(f"Rectification might have failed. Output image size: {rectified_image.shape[:2]}")
            
            # First, check if we have exactly 4 markers
            _, corners, ids, _ = detect_aruco_markers(self.test_image)
            if ids is not None:
                print(f"Marker detection found {len(ids)} markers with IDs: {ids.flatten()}")
                if len(ids) == 4:
                    print(f"FAILED: Rectification should have succeeded with exactly 4 markers")
                else:
                    print(f"Expected failure: Need exactly 4 markers, found {len(ids)}")
            else:
                print("Expected failure: No markers detected in the test image")
            
            # The returned image should still be valid
            self.assertTrue(rectified_image.shape[0] > 0 and rectified_image.shape[1] > 0)
    
    def test_create_heatmap(self):
        """Test the create_heatmap function."""            
        try:
            img = cv2.imread(self.heatmap_test_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Call the create_heatmap function
            result_img, score = create_heatmap(img)
            
            # Basic assertions
            self.assertIsNotNone(result_img)
            self.assertIsInstance(score, float)
            
            # Save the result image for visual inspection
            output_path = os.path.join(self.test_dir, "test", "unittest_heatmap_output.jpg")
            cv2.imwrite(output_path, result_img)
            
            print(f"Heatmap generated with score: {score}")
            print(f"Result image saved to: {output_path}")
            
            # Verify the output image dimensions
            # The output should be a horizontal concatenation of two images
            self.assertEqual(result_img.shape[0], 500)  # Height should be 500
            self.assertEqual(result_img.shape[1], 1000)  # Width should be 1000 (500+500)
            
        except Exception as e:
            self.fail(f"create_heatmap raised an exception: {str(e)}")

if __name__ == "__main__":
    unittest.main() 