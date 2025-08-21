#!/usr/bin/env python3
"""
Test script for the create_heatmap function.
Tests the heatmap generation with different weather scenarios and input images.
"""

import os
import sys
import numpy as np
import cv2 as cv
from datetime import datetime

# Add the current directory to the path to import create_heatmap
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_heatmap import create_heatmap, initialize_resources, matrix_to_heatmap

def load_test_image(image_path):
    """
    Load and convert an image to RGB format.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: RGB image array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Load image using OpenCV (BGR format)
    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return img_rgb

def save_heatmap(heat_matrix, output_path, title="Heatmap"):
    """
    Save the heat matrix as a heatmap using the matrix_to_heatmap function.
    
    Args:
        heat_matrix (np.ndarray): Temperature matrix
        output_path (str): Output file path
        title (str): Title for the file (not used, kept for compatibility)
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert heat matrix to heatmap using the existing function
    heatmap_matrix = matrix_to_heatmap(heat_matrix)
    
    # Save the heatmap as an image
    cv.imwrite(output_path, heatmap_matrix)
    
    print(f"Heatmap saved to: {output_path}")

def run_heatmap_test(image_name, image_path, mode, output_dir):
    """
    Run a single heatmap test for a given image and weather mode.
    
    Args:
        image_name (str): Name of the image (for output naming)
        image_path (str): Path to the input image
        mode (str): Weather mode ('summer_day', 'summer_night', 'real_time')
        output_dir (str): Output directory for results
        
    Returns:
        dict: Test results including score and file paths
    """
    print(f"\n{'='*50}")
    print(f"Testing: {image_name} with {mode} weather")
    print(f"{'='*50}")
    
    try:
        # Load the test image
        img_rgb = load_test_image(image_path)
        print(f"Loaded image: {image_path} (shape: {img_rgb.shape})")
        
        # Generate heatmap
        print(f"Generating heatmap with {mode} weather scenario...")
        heat_matrix, score, weather_data = create_heatmap(img_rgb, mode)
        
        # Print results
        print(f"Temperature Score: {score:.2f}°C")
        print(f"Temperature Range: {np.min(heat_matrix):.2f}°C to {np.max(heat_matrix):.2f}°C")
        print(f"Weather Data: {weather_data['weather']['description']}")
        print(f"Current Temperature: {weather_data['t2m']:.1f}°C")
        print(f"Humidity: {weather_data['rel_humid']:.1f}%")
        print(f"Wind Speed: {weather_data['wind_speed']:.1f} m/s")
        
        # Save heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{image_name}_{mode}_{timestamp}_heatmap.png"
        output_path = os.path.join(output_dir, output_filename)
        
        title = f"Heatmap: {image_name} ({mode})\nScore: {score:.2f}°C"
        save_heatmap(heat_matrix, output_path, title)
        
        return {
            'image_name': image_name,
            'mode': mode,
            'score': score,
            'min_temp': np.min(heat_matrix),
            'max_temp': np.max(heat_matrix),
            'weather_data': weather_data,
            'output_path': output_path,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            'image_name': image_name,
            'mode': mode,
            'error': str(e),
            'success': False
        }

def main():
    """
    Main test function that runs all combinations of images and weather scenarios.
    """
    print("Heatmap Generation Test Suite")
    print("=" * 50)
    
    # Initialize resources (load model and pre-calculate kernels)
    try:
        print("Initializing resources...")
        initialize_resources()
        print("Resources initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize resources: {e}")
        return
    
    # Define test parameters
    test_images = {
        'chaud': '/home/mrcyme/Documents/FARI/POCs/heatmapPOC/backend/test/chaud.jpeg',
        'froid': '/home/mrcyme/Documents/FARI/POCs/heatmapPOC/backend/test/froid.jpeg'
    }
    
    weather_modes = ['summer_day', 'summer_night', 'real_time']
    output_dir = '/home/mrcyme/Documents/FARI/POCs/heatmapPOC/backend/test/output'
    
    # Run all test combinations
    results = []
    
    for image_name, image_path in test_images.items():
        for mode in weather_modes:
            result = run_heatmap_test(image_name, image_path, mode, output_dir)
            results.append(result)
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\nSuccessful Test Results:")
        print(f"{'Image':<10} {'Mode':<15} {'Score':<10} {'Min Temp':<10} {'Max Temp':<10}")
        print("-" * 70)
        for result in successful_tests:
            print(f"{result['image_name']:<10} {result['mode']:<15} "
                  f"{result['score']:<10.2f} {result['min_temp']:<10.2f} {result['max_temp']:<10.2f}")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        for result in failed_tests:
            print(f"- {result['image_name']} ({result['mode']}): {result['error']}")
    
    print(f"\nOutput files saved in: {output_dir}")

if __name__ == "__main__":
    main() 