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
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

# Constants (from constants.py)
WATER = 1
GREEN = 2
IMPERVIOUS = 3
THRESHOLD = 100
EDGE_DETECTION = True
TRANSFORMATIONMATRIX = [[0.99996877, -0.00790621, 107.41512095],
                       [0.00790621, 0.99996877, -8.49420321]]
WIDTH = 16
HEIGHT = 16
GAME_FACTOR = -1

EXTRA_PARAMETERS = {
        "alt": 50,
        "short_wave": 0.0,  # SHORT_WAVE_FROM_SKY_1HOUR
        "t2m": 16.28,  # t2m_inca
        "rel_humid": 82.03,  # rel_humid_inca
        "wind_speed": 1.58,  # wind_speed_inca
        "max_t2m": 33.21,  # max_t2m_inca
        "min_t2m": 15.36,  # min_t2m_inca
        "KERNEL_250M_PX": 1,
        "game_mode": 3,
        "surrounding": 3
    }

# Global variables for min and max values for heat map visualization
global VMIN
global VMAX
VMIN = None
VMAX = None

def set_vminmax(matrix: np.ndarray) -> None:
    """
    Setting VMIN and VMAX to make sure the comparisons use the same color range
    Args:
        matrix: the first matrix to determin the color range MVIN and VMAX
    """
    global VMIN
    VMIN = np.min(matrix)
    global VMAX
    VMAX = np.max(matrix)
    print(f"VMIN: {VMIN}, VMAX: {VMAX}")

def reset_vminmax() -> None:
    """
    Resets the VMIN and VMAX
    """
    global VMIN
    VMIN = None
    global VMAX
    VMAX = None
    print(f"VMIN: {VMIN}, VMAX: {VMAX}")

# From image_preprocessing.py
def find_contour(src_gray: np.ndarray, threshold: int) -> np.ndarray:
    """
    Find contour in the image
    Args:
        src_gray (np.ndarray): Gray scale image
        threshold (int): Threshold

    Returns:
        np.ndarray: Contour that outlines the city
    """
    canny_output: np.ndarray = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=lambda x: cv.arcLength(x, True))

    return largest_contour

def find_transformation_matrix(img: np.ndarray) -> (np.ndarray, int, int):
    """
    Find the transformation matrix to rotate and translate the image to center the city
    Args:
        img (np.ndarray): Image to be transformed

    Returns:
        np.ndarray: Transformation matrix
        int: Width of the city in the image
        int: Height of the city in the image
    """
    # Convert image to gray and blur it
    src_gray: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    src_gray: np.ndarray = cv.blur(src_gray, (3, 3))
    threshold: int = THRESHOLD

    contour = find_contour(src_gray, threshold)

    # Find the rotated rectangles for each contour
    min_rect: tuple[list[float], list[int], float] = cv.minAreaRect(contour)

    box: np.ndarray = cv.boxPoints(min_rect)
    box: np.intp = np.intp(box)

    point0: tuple = box[0]
    point2: tuple = box[2]

    middle_x: int = (point0[0] + point2[0]) / 2
    middle_y: int = (point0[1] + point2[1]) / 2

    center: tuple = (math.floor(middle_x), math.floor(middle_y))
    width: float = math.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2)
    height: float = math.sqrt((box[2][0] - box[1][0]) ** 2 + (box[2][1] - box[1][1]) ** 2)

    # calculate the image rotation using inverse tan
    divider: float = box[1][1] - box[0][1]
    ang: float = 0
    if divider != 0:
        ang: float = math.atan((box[1][0] - box[0][0]) / divider)  # in radians

    ang_degrees: float = math.degrees(ang)  # in degrees
    transformation_matrix: np.ndarray = cv.getRotationMatrix2D(center, -ang_degrees + 180, scale=1.0)

    # Calculate the translation needed to center the square
    tx: float = center[0] - width / 2
    ty: float = center[1] - height / 2
    # Add translation to Transformation Matrix
    transformation_matrix[0, 2] -= tx
    transformation_matrix[1, 2] -= ty

    return transformation_matrix, width, height

def transform_img(img: np.ndarray) -> np.ndarray:
    """
    Transform an image by applying the transformation matrix
    Args:
        img (np.ndarray): Image to be transformed

    Returns:
        np.ndarray: Transformed image
    """
    if EDGE_DETECTION:
        m, w, h = find_transformation_matrix(img)
    else:
        m, w, h = np.array(TRANSFORMATIONMATRIX), WIDTH, HEIGHT
    return cv.warpAffine(src=img, M=m, dsize=(math.floor(w), math.floor(h)))

def img_to_matrix(img: np.ndarray) -> np.ndarray:
    """
    Transform an image to a matrix
    Args:
        img (np.ndarray): Image to be processed

    Returns:
        np.ndarray: Transformed image
    """
    # Convert image to HSV format
    img: np.ndarray = img.astype(np.uint8)
    hsv_image: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    height, width, _ = hsv_image.shape

    matrix: np.ndarray = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # h-s-v-values between 0 - 255
            h, s, v = hsv_image[y, x]
            if h < 30 and s > 127 and v > 100:
                matrix[y, x] = WATER  # Blue
            elif 35 < h < 85 and s > 127 and v > 50:
                matrix[y, x] = GREEN  # Green
            else:
                matrix[y, x] = IMPERVIOUS  # Black

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
    Converts a matrix to a heatmap
    Args:
        matrix (np.ndarray): Matrix to be converted

    Returns:
        np.ndarray: Heatmap
    """
    plt.imshow(matrix.T, cmap='plasma', interpolation='bilinear', vmin=VMIN, vmax=VMAX)
    plt.axis('off')
    plt.colorbar()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Read the image from the BytesIO object using PIL
    heatmap_image = Image.open(buffer)

    # Convert PIL Image to NumPy array
    heatmap_array = np.array(heatmap_image)

    # Clear the plot to avoid displaying it in the console
    plt.clf()

    # Return the heatmap array
    return heatmap_array

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
    Create the kernel of specified size
    Kernel values are calculated by the following formula:
        create a larger matrix (10x)
        calculate which points are in or outside the radius
        assign 0s and 1s accordingly
        now calculate the kernel by calculating the coverage percentage for each sector
    This gives an approximation for a circular kernel
    Args:
        kernel_size (int): Size of the kernel

    Returns:
        list[list[int]]: Kernel of specified size
    """
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
    Create a matrix containing the temperatures for each area
    Args:
        matrix (np.ndarray): Matrix containing the ground type information
        surrounding_category (int): Specifies the surrounding to be used for the heatmap
        extra_parameters (dict): Additional parameters for the AI-model

    Returns:
        np.ndarray: Heat matrix of the provided matrix
    """
    kernel_250m_px = extra_parameters["KERNEL_250M_PX"]
    kernel_250m = create_kernel_of_size(kernel_250m_px)
    kernel_1km = create_kernel_of_size(kernel_250m_px * 4 - 1)
    kernel_sum_250m = np.sum(np.array(kernel_250m))
    kernel_sum_1km = np.sum(np.array(kernel_1km))
    digit_matrix = matrix_to_digit_matrix(matrix)

    # 250m Matrix
    matrix_kernel_250m = apply_kernel(digit_matrix, kernel_250m, surrounding_category)
    water_matrix_250m, green_matrix_250m, impervious_matrix_250m = split_matrix(matrix_kernel_250m)
    water_float_matrix_250m = water_matrix_250m / float(kernel_sum_250m)
    green_float_matrix_250m = green_matrix_250m / float(kernel_sum_250m)
    impervious_float_matrix_250m = impervious_matrix_250m / float(kernel_sum_250m)

    # 1km Matrix
    matrix_kernel_1km = apply_kernel(digit_matrix, kernel_1km, surrounding_category)
    water_matrix_1km, green_matrix_1km, impervious_matrix_1km = split_matrix(matrix_kernel_1km)
    water_float_matrix_1km = water_matrix_1km / float(kernel_sum_1km)
    green_float_matrix_1km = green_matrix_1km / float(kernel_sum_1km)
    impervious_float_matrix_1km = impervious_matrix_1km / float(kernel_sum_1km)
    
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
        'ALT': [extra_parameters["alt"]] * total_pixels,
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

    # Load the model and make predictions
    model = load("random_forest.joblib")
    predictions = model.predict(df)

    # Reshape the predictions to the original matrix shape
    heat_matrix = predictions.reshape(shape)
    return heat_matrix

def process_img(img: np.ndarray, extra_parameters: dict) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to process an image
    Args:
        img (np.ndarray): Image to be processed
        extra_parameters (dict): Extra parameters for the AI_model

    Returns:
        np.ndarray: Processed image
        np.ndarray: Heatmap
        np.ndarray: Heat matrix
    """
    #img_transform = transform_img(img)
    img_compress = compress_img(img, 16)
    matrix = img_to_matrix(img_compress)
    heat_matrix = create_heatmatrix_from_matrix(matrix, extra_parameters["surrounding"], extra_parameters)
    heatmap = matrix_to_heatmap(heat_matrix)
    return img, heatmap, heat_matrix

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

# Modified function to save the image without API call
def save_image(filename: str, img: np.ndarray) -> str:
    """
    Save image to file
    Args:
        filename (str): Filename to save the image as
        img (np.ndarray): Image to save

    Returns:
        str: Path to the saved image
    """
    path = f"{filename}.png"
    cv.imwrite(path, img)
    return path

# Game mode 3 function - modified to save locally instead of using API
def game_mode_3(src_img: np.ndarray, extra_parameters: dict) -> [np.ndarray, np.ndarray, str, float]:
    """
    Processing image for game mode 3
    Args:
        src_img (np.ndarray): Image of the city
        extra_parameters (dict): Extra parameters for the AI-model

    Returns:
        np.ndarray: Cut image of the city
        np.ndarray: Heatmap of the city
        str: Path to saved image
        float: Score
    """
    reset_vminmax()
    src_img_cut, src_heatmap, src_heat_matrix = process_img(src_img, extra_parameters)

    src_img_out = cv.cvtColor(enlarge_img(src_img_cut, 500), cv.COLOR_RGB2BGR)
    src_heat_out = cv.cvtColor(enlarge_img(src_heatmap, 500), cv.COLOR_RGBA2BGR)
    cv.imwrite("src_img_out.jpg", src_img_out)
    cv.imwrite("src_heat_out.jpg", src_heat_out)
    score = calculate_score(src_heat_matrix)

    # Concatenate images horizontally
    result_image = np.concatenate((src_img_out, src_heat_out), axis=1)

    # Save the image locally
    path = save_image("3mode_result", result_image)
    
    return src_img_cut, src_heatmap, path, score


def create_heatmap(src_img: np.ndarray) -> [np.ndarray, float]:
    """
    Create a heatmap from an image
    Args:
        src_img (np.ndarray): Image of the city

    Returns:
    """
    
    if src_img is None:
        print(f"Error: Could not load image from {input_image_path}")
        exit(1)
    
    # Execute game mode 3
    src_img_cut, src_heatmap, result_path, score = game_mode_3(src_img, EXTRA_PARAMETERS)
    
    print(f"Processing complete!")
    print(f"Result image saved to: {result_path}")
    print(f"City heat score: {score}")
    
    # Optionally display the result image
    result_img = cv.imread(result_path)
    return result_img, score