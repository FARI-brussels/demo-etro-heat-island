"""
Standalone script to execute game_mode_3 function.
This script extracts only the necessary code from the original codebase.
"""
import math
import numpy as np
import cv2 as cv
import pandas as pd
from joblib import load
from scipy.signal import convolve2d
import requests
import json

# Constants (from constants.py)
WATER = 1
GREEN = 2
IMPERVIOUS = 3
WIDTH = 16
HEIGHT = 16


def fetch_weather_data():
    """
    Fetches weather data from the Brussels Mobility Twin API and converts temperatures from Kelvin to Celsius.
    Returns a dictionary with the processed weather data.
    """
    url = "https://api.mobilitytwin.brussels/environment/weather"
    api_key = "6bda3e364cf545f2f8a93340dc0e99e6ad82e43010074868b9fc7c02cc30d86eb9f9b52a543d3eed6f04f12505614cc1bf5ae2b57d04123d4931c34f5ef31eec"
    
    try:
        response = requests.get(url, headers={
            'Authorization': f'Bearer {api_key}'
        })
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Convert temperatures from Kelvin to Celsius
        temp_celsius = data['main']['temp'] - 273.15
        temp_min_celsius = data['main']['temp_min'] - 273.15
        temp_max_celsius = data['main']['temp_max'] - 273.15
        
        return {
            't2m': temp_celsius,
            'max_t2m': temp_max_celsius,
            'min_t2m': temp_min_celsius,
            'rel_humid': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        # Return default values if API call fails
        return {
            't2m': 16.28,
            'max_t2m': 33.21,
            'min_t2m': 15.36,
            'rel_humid': 82.03,
            'wind_speed': 1.58
        }

weather_data = fetch_weather_data()
EXTRA_PARAMETERS = {
        "alt": 50,
        "short_wave": 0.0,  # SHORT_WAVE_FROM_SKY_1HOUR
            "t2m": weather_data['t2m'],
            "rel_humid": weather_data['rel_humid'],
            "wind_speed": weather_data['wind_speed'],
            "max_t2m": weather_data['max_t2m'],
            "min_t2m": weather_data['min_t2m'],
        "KERNEL_250M_PX": 1,
        "game_mode": 3,
        "surrounding": 3
    }
# Global variables for preloaded resources
PRELOADED_MODEL = None
PRECALCULATED_KERNEL_250M = None
PRECALCULATED_KERNEL_1KM = None
PRECALCULATED_KERNEL_SUM_250M = None
PRECALCULATED_KERNEL_SUM_1KM = None

# Global variables for min and max values for heat map visualization (can be set if needed)
VMIN = None  # Example: 15
VMAX = None  # Example: 35

def initialize_resources():
    """Loads the model and pre-calculates kernels."""
    global PRELOADED_MODEL, PRECALCULATED_KERNEL_250M, PRECALCULATED_KERNEL_1KM
    global PRECALCULATED_KERNEL_SUM_250M, PRECALCULATED_KERNEL_SUM_1KM

    print("Initializing resources for heatmap generation...")
    # Load the model
    try:
        PRELOADED_MODEL = load("random_forest.joblib")
        print("Random Forest model loaded successfully.")
    except FileNotFoundError:
        print("ERROR: random_forest.joblib not found. Make sure the model file is in the correct path.")
        raise
    except Exception as e:
        print(f"ERROR: Could not load random_forest.joblib: {e}")
        raise

    # Pre-calculate kernels
    kernel_250m_px = EXTRA_PARAMETERS["KERNEL_250M_PX"]
    # Ensure kernel_250m_px is odd for create_kernel_of_size
    if kernel_250m_px % 2 == 0:
        print(f"Warning: KERNEL_250M_PX ({kernel_250m_px}) is even. Adjusting to {kernel_250m_px + 1} for kernel creation.")
        kernel_250m_px += 1
    
    kernel_1km_px = kernel_250m_px * 4 - 1 # This might result in an even number.
    if kernel_1km_px % 2 == 0: # Ensure 1km kernel is also odd
        print(f"Warning: Calculated 1km kernel size ({kernel_1km_px}) is even. Adjusting to {kernel_1km_px + 1}.")
        kernel_1km_px +=1


    PRECALCULATED_KERNEL_250M = create_kernel_of_size(kernel_250m_px)
    PRECALCULATED_KERNEL_1KM = create_kernel_of_size(kernel_1km_px)
    PRECALCULATED_KERNEL_SUM_250M = np.sum(np.array(PRECALCULATED_KERNEL_250M))
    PRECALCULATED_KERNEL_SUM_1KM = np.sum(np.array(PRECALCULATED_KERNEL_1KM))
    print("Kernels pre-calculated successfully.")
    print(f"Kernel 250m size: {kernel_250m_px}, sum: {PRECALCULATED_KERNEL_SUM_250M}")
    print(f"Kernel 1km size: {kernel_1km_px}, sum: {PRECALCULATED_KERNEL_SUM_1KM}")


def img_to_matrix(img_rgb: np.ndarray) -> np.ndarray:
    """
    Transform an RGB image to a matrix of categories (WATER, GREEN, IMPERVIOUS).
    Optimized using vectorized operations.
    Args:
        img_rgb (np.ndarray): RGB Image to be processed (height, width, 3)

    Returns:
        np.ndarray: Transformed matrix (height, width) with values 1, 2, or 3.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Input image must be an RGB image (height, width, 3).")

    hsv_image = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV_FULL)
    height, width, _ = hsv_image.shape
    matrix = np.full((height, width), IMPERVIOUS, dtype=np.uint8) # Default to IMPERVIOUS

    # Define HSV ranges for water (Blue in original logic)
    # H < 30, S > 127, V > 100
    # OpenCV HSV: H [0-255], S [0-255], V [0-255] for COLOR_RGB2HSV_FULL
    lower_water = np.array([0, 128, 101])
    upper_water = np.array([29, 255, 255]) # H < 30
    water_mask = cv.inRange(hsv_image, lower_water, upper_water)
    matrix[water_mask > 0] = WATER

    # Define HSV ranges for green
    # 35 < H < 85, S > 127, V > 50
    lower_green = np.array([36, 128, 51]) # H > 35
    upper_green = np.array([84, 255, 255]) # H < 85
    green_mask = cv.inRange(hsv_image, lower_green, upper_green)
    matrix[green_mask > 0] = GREEN

    return matrix

# From image_processing_utils.py
def compress_img(img: np.ndarray, size: int) -> np.ndarray:
    """
    Compress image to specified size
    Args:
        img (np.ndarray): Image to be compressed
        size (int): Target size

    Returns:
        np.ndarray: Compressed image
    """
    return cv.resize(img, (size, size))

def enlarge_img(img: np.ndarray, size: int) -> np.ndarray:
    """
    Enlarge image to specified size
    Args:
        img (np.ndarray): Image to be enlarged
        size (int): Target size

    Returns:
        np.ndarray: Enlarged image
    """
    return cv.resize(img, (size, size), interpolation=cv.INTER_NEAREST)

def matrix_to_digit_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a matrix
    Original matrix represents the ground type by 1, 2, 3
    New matrix represents the ground type by 1, 1_000, 1_000_000
    Args:
        matrix (np.ndarray): Matrix to be converted

    Returns:
        np.ndarray: Converted matrix
    """
    # Create a copy of the matrix to avoid modifying the original
    matrix_edit: np.ndarray = np.copy(matrix)

    # Ensure int is used not uint8
    matrix_edit = matrix_edit.astype(int)

    # Replace every 2 with 1_000
    matrix_edit[matrix_edit == 2] = 1_000

    # Replace every 3 with 1_000_000
    matrix_edit[matrix_edit == 3] = 1_000_000

    return matrix_edit

def split_matrix(src_matrix: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a matrix into three different matrices
    Matrix encoding all three surfaces is split
        using integer division and modulo operations into three matrices
    Args:
        src_matrix (np.ndarray): Matrix to be split

    Returns:
        np.ndarray: Matrix containing the information for water
        np.ndarray: Matrix containing the information for plants
        np.ndarray: Matrix containing the information for impervious
    """
    water_matrix = np.mod(src_matrix, 1_000)
    green_matrix = np.mod(np.floor_divide(src_matrix, 1_000), 1_000)
    impervious_matrix = np.floor_divide(src_matrix, 1_000_000)
    return water_matrix, green_matrix, impervious_matrix

def matrix_to_heatmap(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a matrix of temperature values to a normalized matrix (0-255 range).
    Args:
        matrix (np.ndarray): Matrix of temperature values to be converted.

    Returns:
        np.ndarray: Normalized matrix (0-255 range) for heatmap visualization.
    """
    if matrix is None or matrix.size == 0:
        raise ValueError("Input matrix for heatmap generation is empty or None.")

    # Normalize matrix to 0-255 range for colormap application
    min_val = VMIN if VMIN is not None else np.min(matrix)
    max_val = VMAX if VMAX is not None else np.max(matrix)
    
    if max_val == min_val: # Avoid division by zero if all values are the same
        normalized_matrix = np.zeros_like(matrix, dtype=np.uint8)
    else:
        normalized_matrix = (255 * (matrix - min_val) / (max_val - min_val)).astype(np.uint8)
    return normalized_matrix

# From image_processing.py
def apply_kernel(matrix: np.ndarray, kernel: list[list[int]], fillvalue: int) -> np.ndarray:
    """
    Apply the given kernel to the matrix
    Args:
        matrix (np.ndarray): the matrix to apply the kernel to
        kernel (list[list[int]]): the kernel to use
        fillvalue (int): the value to use for padding

    Returns:
        np.ndarray: Resulting matrix
    """
    transposed_matrix = convolve2d(matrix, kernel, mode='same', boundary='fill', fillvalue=fillvalue)
    result_matrix = transposed_matrix.T
    return result_matrix

def create_kernel_of_size(kernel_size: int) -> list[list[int]]:
    """
    Create the kernel of specified size.
    Kernel values are calculated by the following formula:
        create a larger matrix (10x)
        calculate which points are in or outside the radius
        assign 0s and 1s accordingly
        now calculate the kernel by calculating the coverage percentage for each sector
    This gives an approximation for a circular kernel.
    Args:
        kernel_size (int): Size of the kernel. Must be an odd number.

    Returns:
        list[list[int]]: Kernel of specified size.
    """
    if kernel_size % 2 == 0:
        # This check should ideally be handled before calling, 
        # but as a safeguard:
        print(f"Error: Kernel size must be an odd number. Received {kernel_size}.")
        # Defaulting to a minimally invasive change if an even number is passed.
        # Proper handling should ensure odd numbers are passed from the caller.
        kernel_size +=1 
        print(f"Adjusted kernel size to {kernel_size}")


    scalar = 11
    scaled_kernel_size = kernel_size * scalar
    scaled_kernel = [[0] * scaled_kernel_size for _ in range(scaled_kernel_size)]

    if kernel_size % 2 == 0:
        raise ValueError("Scaled kernel size must be an odd number.")

    center = scaled_kernel_size // 2

    for i in range(scaled_kernel_size):
        for j in range(scaled_kernel_size):
            distance = math.sqrt((j - center) ** 2 + (i - center) ** 2)
            if distance < center:
                scaled_kernel[i][j] = 1

    downsized_kernel_size = kernel_size
    downsized_kernel = [[0] * downsized_kernel_size for _ in range(downsized_kernel_size)]

    for i in range(downsized_kernel_size):
        for j in range(downsized_kernel_size):
            subfield_sum = 0
            for m in range(i * scalar, (i + 1) * scalar):
                for n in range(j * scalar, (j + 1) * scalar):
                    subfield_sum += scaled_kernel[m][n]
            downsized_kernel[i][j] = subfield_sum

    return downsized_kernel

def create_heatmatrix_from_matrix(matrix: np.ndarray, surrounding_category: int, extra_parameters: dict) -> np.ndarray:
    """
    Create a matrix containing the temperatures for each area using preloaded model and kernels.
    Args:
        matrix (np.ndarray): Matrix containing the ground type information
        surrounding_category (int): Specifies the surrounding to be used for the heatmap
        extra_parameters (dict): Additional parameters for the AI-model (uses global EXTRA_PARAMETERS)

    Returns:
        np.ndarray: Heat matrix of the provided matrix
    """
    if PRELOADED_MODEL is None or PRECALCULATED_KERNEL_250M is None or PRECALCULATED_KERNEL_1KM is None:
        raise RuntimeError("Resources not initialized. Call initialize_resources() first.")

    digit_matrix = matrix_to_digit_matrix(matrix)

    # 250m Matrix
    matrix_kernel_250m = apply_kernel(digit_matrix, PRECALCULATED_KERNEL_250M, surrounding_category)
    water_matrix_250m, green_matrix_250m, impervious_matrix_250m = split_matrix(matrix_kernel_250m)
    water_float_matrix_250m = water_matrix_250m / float(PRECALCULATED_KERNEL_SUM_250M)
    green_float_matrix_250m = green_matrix_250m / float(PRECALCULATED_KERNEL_SUM_250M)
    impervious_float_matrix_250m = impervious_matrix_250m / float(PRECALCULATED_KERNEL_SUM_250M)

    # 1km Matrix
    matrix_kernel_1km = apply_kernel(digit_matrix, PRECALCULATED_KERNEL_1KM, surrounding_category)
    water_matrix_1km, green_matrix_1km, impervious_matrix_1km = split_matrix(matrix_kernel_1km)
    water_float_matrix_1km = water_matrix_1km / float(PRECALCULATED_KERNEL_SUM_1KM)
    green_float_matrix_1km = green_matrix_1km / float(PRECALCULATED_KERNEL_SUM_1KM)
    impervious_float_matrix_1km = impervious_matrix_1km / float(PRECALCULATED_KERNEL_SUM_1KM)
    
    # Flatten the matrices to process in batches
    shape = water_float_matrix_250m.shape
    total_pixels = np.prod(shape)
    flat_waters_250m = water_float_matrix_250m.flatten()
    flat_greens_250m = green_float_matrix_250m.flatten()
    flat_impervious_250m = impervious_float_matrix_250m.flatten()
    flat_waters_1km = water_float_matrix_1km.flatten()
    flat_greens_1km = green_float_matrix_1km.flatten()
    flat_impervious_1km = impervious_float_matrix_1km.flatten()

    # Prepare the input DataFrame
    df = pd.DataFrame({
        'ALT': [extra_parameters["alt"]] * total_pixels, # Use global EXTRA_PARAMETERS
        'WATER': flat_waters_250m,
        'GREEN': flat_greens_250m,
        'IMPERVIOUS': flat_impervious_250m,
        'WATER_1000': flat_waters_1km,
        'GREEN_1000': flat_greens_1km,
        'IMPERVIOUS_1000': flat_impervious_1km,
        'SHORT_WAVE_FROM_SKY_1HOUR': [extra_parameters["short_wave"]] * total_pixels,
        't2m_inca': [extra_parameters["t2m"]] * total_pixels,
        'rel_humid_inca': [extra_parameters["rel_humid"]] * total_pixels,
        'wind_speed_inca': [extra_parameters["wind_speed"]] * total_pixels,
        'max_t2m_inca': [extra_parameters["max_t2m"]] * total_pixels,
        'min_t2m_inca': [extra_parameters["min_t2m"]] * total_pixels
    })

    # Use the preloaded model
    predictions = PRELOADED_MODEL.predict(df)

    # Reshape the predictions to the original matrix shape
    heat_matrix = predictions.reshape(shape)
    return heat_matrix

def process_img(img_rgb: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to process an RGB image. Uses global EXTRA_PARAMETERS.
    Args:
        img_rgb (np.ndarray): RGB Image to be processed

    Returns:
        np.ndarray: Original processed (resized) RGB image
        np.ndarray: Heatmap RGB image
        np.ndarray: Heat matrix
    """
    cv.imwrite("image_in.png", img_rgb)
    # Assuming img_rgb is already in the correct color format (RGB)
    # No transformation like transform_img is called here.
    img_compress = compress_img(img_rgb, WIDTH) # Use WIDTH, HEIGHT constants
    
    matrix = img_to_matrix(img_compress) # Expects RGB
    weather_data = fetch_weather_data()

    EXTRA_PARAMETERS = {
            "alt": 50,
            "short_wave": 0.0,  # SHORT_WAVE_FROM_SKY_1HOUR
            "t2m": weather_data['t2m'],
            "rel_humid": weather_data['rel_humid'],
            "wind_speed": weather_data['wind_speed'],
            "max_t2m": weather_data['max_t2m'],
            "min_t2m": weather_data['min_t2m'],
            "KERNEL_250M_PX": 1,
            "game_mode": 3,
            "surrounding": 3
        }

    # Use global EXTRA_PARAMETERS for surrounding category and other model params
    heat_matrix = create_heatmatrix_from_matrix(matrix, EXTRA_PARAMETERS["surrounding"], EXTRA_PARAMETERS)
    
    return heat_matrix, weather_data # Return RGB images

def calculate_score(src_heat_matrix: np.ndarray) -> float:
    """
    Calculates the score of the heat matrix
    Args:
        src_heat_matrix (np.ndarray): Heat matrix

    Returns:
        float: The score (mean) of the heat matrix
    """
    score = np.mean(src_heat_matrix)
    return score




def create_heatmap(src_img_rgb: np.ndarray) -> [np.ndarray, float, float, float]:
    """
    Create a heatmap from an RGB image.
    Args:
        src_img_rgb (np.ndarray): RGB Image of the city

    Returns:
        np.ndarray: Normalized matrix for heatmap visualization.
        float: Temperature score.
        float: Minimum temperature value.
        float: Maximum temperature value.
    """
    if src_img_rgb is None:
        print(f"Error: Input image for create_heatmap is None.")
        raise ValueError("Input image cannot be None.") 
    
    # Execute game mode 3, which now uses global EXTRA_PARAMETERS
    src_heat_matrix, weather_data = process_img(src_img_rgb)
    

    
    # Calculate score
    score = calculate_score(src_heat_matrix)
    
    return src_heat_matrix , score, weather_data