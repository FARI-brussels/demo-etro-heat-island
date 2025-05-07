from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import cv2 as cv
import time
import threading
from create_heatmap import create_heatmap

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Request tracking mechanism
active_requests = {}
request_lock = threading.Lock()
current_task = None
current_task_lock = threading.Lock()

def is_newest_request(request_id):
    """Check if a request is the newest one"""
    with request_lock:
        if not active_requests:
            return False
        newest_id = max(active_requests.items(), key=lambda x: x[1])[0]
        return request_id == newest_id

def register_request():
    """Register a new request with current timestamp and return a unique ID"""
    request_id = str(time.time())
    with request_lock:
        active_requests[request_id] = time.time()
    return request_id

def remove_request(request_id):
    """Remove a request from tracking"""
    with request_lock:
        if request_id in active_requests:
            del active_requests[request_id]

def clean_old_requests(except_id=None):
    """Clean all requests except the specified one"""
    with request_lock:
        request_ids = list(active_requests.keys())
        for req_id in request_ids:
            if except_id is None or req_id != except_id:
                del active_requests[req_id]

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Register this request
        request_id = register_request()
        
        # Get the base64 encoded image from the request
        data = request.json
        if not data or 'image' not in data:
            remove_request(request_id)
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Check if this is still the newest request before heavy processing
        if not is_newest_request(request_id):
            remove_request(request_id)
            return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200
        
        # Set current task
        global current_task
        with current_task_lock:
            current_task = request_id
        
        # Process the image - first detect ArUco markers
        processed_image = crop_and_rectify_aruco_square(image_np)
        if processed_image is None:
            remove_request(request_id)
            return jsonify({'error': 'Could not find 4 ArUco markers'}), 400
        
        # Check again if this is still the newest request
        if not is_newest_request(request_id):
            remove_request(request_id)
            return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200
        
        # Generate heatmap
        result_img, score = create_heatmap(processed_image)
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(result_img)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        processed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Clean up and clear tracking
        with current_task_lock:
            current_task = None
        clean_old_requests()
        
        return jsonify({
            'status': 'success',
            'processed_image': processed_base64, 
            'temperature': score
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        # Make sure to clean up
        if 'request_id' in locals():
            remove_request(request_id)
        return jsonify({'error': str(e)}), 500

def detect_aruco_markers(image):
    """
    Detect ArUco markers in the image.
    If at least 4 markers are found, highlight them in the image.
    """
    # Convert to grayscale for marker detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Detect ArUco markers
    # Using DICT_4X4_50 as in the user's provided function
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    # Return the output image and the detection results
    return corners, ids, rejected

def order_points(pts):
    # Sort points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # Now, sort the left-most coordinates by their y-coordinates so we can
    # grab the top-left and bottom-left points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # Now, perform the same for the right-most points to obtain the
    # top-right and bottom-right points
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    # Return the coordinates in top-left, top-right, bottom-right, bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def crop_and_rectify_aruco_square(image, target_size=(300, 300)):
    """
    Detects 4 ArUco markers positioned at the corners of a square region.
    Specifically selects the inner corner of each marker to define the region to rectify.
    
    For markers arranged in a square:
    - Top-left marker: use bottom-right corner
    - Top-right marker: use bottom-left corner
    - Bottom-right marker: use top-left corner
    - Bottom-left marker: use top-right corner

    Args:
        image: The input image (NumPy array).
        target_size: The desired size (width, height) of the output square image.

    Returns:
        The rectified square image (NumPy array) if successful, or the original
        image with an error message drawn on it if 4 markers are not detected
        or processing fails.
    """
    # First, detect the markers using the provided function
    corners, ids, rejected = detect_aruco_markers(image) # Use a copy to avoid modifying the original input

    # Check if exactly 4 markers are detected
    if ids is None or len(ids) != 4:
        return None

    try:
        # Calculate the center of each marker
        marker_centers = []
        for i, corner in enumerate(corners):
            # Each corner is an array of 4 points (the corners of the marker)
            # Calculate the center as the mean of these points
            center = np.mean(corner.squeeze(), axis=0)
            marker_centers.append((center, i))  # Store the center with the marker index
        
        # Sort markers based on their position in the image
        # First, find the center of all the markers
        centroid = np.mean([mc[0] for mc in marker_centers], axis=0)
        
        # Classify markers into quadrants relative to the centroid
        top_left = None
        top_right = None
        bottom_right = None
        bottom_left = None
        
        for center, idx in marker_centers:
            # Determine which quadrant this marker is in
            if center[0] < centroid[0] and center[1] < centroid[1]:
                top_left = idx
            elif center[0] > centroid[0] and center[1] < centroid[1]:
                top_right = idx
            elif center[0] > centroid[0] and center[1] > centroid[1]:
                bottom_right = idx
            elif center[0] < centroid[0] and center[1] > centroid[1]:
                bottom_left = idx
        
        # Verify we found a marker in each quadrant
        if None in [top_left, top_right, bottom_right, bottom_left]:
            return None
        
        # For each marker, extract the appropriate corner based on its position:
        quad_corners = np.zeros((4, 2), dtype=np.float32)
        
        # Extract corners from each ArUco marker
        # ArUco marker corners are ordered: top-left, top-right, bottom-right, bottom-left
        
        # Top-left marker: use bottom-right corner (index 2)
        tl_marker_corners = corners[top_left].squeeze()
        quad_corners[0] = tl_marker_corners[2]  # Bottom-right corner
        
        # Top-right marker: use bottom-left corner (index 3)
        tr_marker_corners = corners[top_right].squeeze()
        quad_corners[1] = tr_marker_corners[3]  # Bottom-left corner
        
        # Bottom-right marker: use top-left corner (index 0)
        br_marker_corners = corners[bottom_right].squeeze()
        quad_corners[2] = br_marker_corners[0]  # Top-left corner
        
        # Bottom-left marker: use top-right corner (index 1)
        bl_marker_corners = corners[bottom_left].squeeze()
        quad_corners[3] = bl_marker_corners[1]  # Top-right corner
        
        # Define the destination points for the rectified square
        (width, height) = target_size
        dst_pts = np.array([
            [0, 0],                 # Top-left
            [width - 1, 0],         # Top-right
            [width - 1, height - 1], # Bottom-right
            [0, height - 1]          # Bottom-left
        ], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(quad_corners, dst_pts)

        # Apply the perspective transformation to rectify the image
        rectified_image = cv2.warpPerspective(image, M, target_size)
        
        return rectified_image
        
    except Exception as e:
        # If any error occurs during processing, log it and return the output image with error message
        print(e)


@app.route('/cancel_processing', methods=['POST'])
def cancel_processing():
    """Endpoint to cancel any ongoing processing"""
    with current_task_lock:
        current_task = None
    
    clean_old_requests()
    return jsonify({'status': 'success', 'message': 'All processing cancelled'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 