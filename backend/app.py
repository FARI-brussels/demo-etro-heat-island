from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import cv2 as cv # This import seems redundant if cv2 is already imported
import time
import threading
# Assuming create_heatmap is defined in create_heatmap.py
from create_heatmap import create_heatmap

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Variable to store the ID of the latest request received
latest_request_id = None
# Lock to protect access to the latest_request_id variable
latest_request_lock = threading.Lock()

# Lock to ensure only one thread executes the heavy image processing at a time
processing_lock = threading.Lock()

# --- Helper functions (kept from your original code) ---

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

    # Return the detection results
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
        The rectified square image (NumPy array) if successful, or None
        if 4 markers are not detected or processing fails.
    """
    # First, detect the markers using the provided function
    corners, ids, rejected = detect_aruco_markers(image)

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
            [0, 0],             # Top-left
            [width - 1, 0],     # Top-right
            [width - 1, height - 1], # Bottom-right
            [0, height - 1]      # Bottom-left
        ], dtype="float32")

        # Get the perspective transformation matrix
        M = cv2.getPerspectiveTransform(quad_corners, dst_pts)

        # Apply the perspective transformation to rectify the image
        rectified_image = cv2.warpPerspective(image, M, target_size)

        return rectified_image

    except Exception as e:
        # If any error occurs during processing, log it and return None
        print(f"Error in crop_and_rectify_aruco_square: {e}")
        return None

# --- Flask Routes ---

@app.route('/process_image', methods=['POST'])
def process_image():
    global latest_request_id
    # Generate a unique ID for this request (timestamp is simple)
    request_id = str(time.time())

    # Update the latest request ID. This is done quickly under a small lock.
    with latest_request_lock:
        latest_request_id = request_id
        print(f"Registered request: {request_id}. Latest is now: {latest_request_id}")


    # Acquire the processing lock. If another request is currently processing,
    # this thread will wait here until the lock is released.
    # Only one thread can hold this lock at a time.
    with processing_lock:
        # --- Critical Processing Section ---
        # Now that we have the processing lock, check if this request is still
        # the latest one. A newer request might have arrived while we were
        # waiting for the lock.
        with latest_request_lock:
            if request_id != latest_request_id:
                print(f"Request {request_id} is not the latest ({latest_request_id}). Cancelling.")
                # Release the processing_lock automatically upon exiting the 'with' block
                # Return a cancellation response
                return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200

        # If we reach here, this request is the latest and has the processing lock.
        print(f"Proceeding with processing for request: {request_id}")

        try:
            # Get the base64 encoded image from the request
            data = request.json
            if not data or 'image' not in data:
                 # Check if still latest before returning error
                with latest_request_lock:
                    if request_id != latest_request_id:
                        print(f"Request {request_id} cancelled during data check by newer request {latest_request_id}.")
                        return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200
                return jsonify({'error': 'No image data provided'}), 400

            # Decode the base64 image
            image_data = base64.b64decode(data['image'])

            # Convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)

            # Process the image - first detect ArUco markers
            processed_image = crop_and_rectify_aruco_square(image_np)
            if processed_image is None:
                 # Check if still latest before returning error
                with latest_request_lock:
                    if request_id != latest_request_id:
                         print(f"Request {request_id} cancelled during ArUco detection by newer request {latest_request_id}.")
                         return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200
                return jsonify({'error': 'Could not find 4 ArUco markers'}), 400

            # Check again if this is still the newest request before the next heavy step
            with latest_request_lock:
                if request_id != latest_request_id:
                     print(f"Request {request_id} cancelled before heatmap by newer request {latest_request_id}.")
                     return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200

            # Generate heatmap - This is the function that will only run one instance at a time
            # because it's inside the 'processing_lock' block.
            print(f"Calculating heatmap for request: {request_id}")
            result_img, score = create_heatmap(processed_image)
            print(f"Heatmap calculated for request: {request_id}")


            # Convert to PIL Image and then to base64
            pil_image = Image.fromarray(result_img)
            buffer = io.BytesIO()
            # Use JPEG format for smaller size, adjust quality as needed
            pil_image.save(buffer, format='JPEG', quality=85)
            processed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Processing is complete for the latest request that acquired the lock.
            # The next request will overwrite latest_request_id if it arrives.

            print(f"Successfully processed request {request_id} and sending response.")
            return jsonify({
                'status': 'success',
                'processed_image': processed_base64,
                'temperature': score
            })

        except Exception as e:
            print(f"Error processing image for request {request_id}: {e}")
            # Check if still latest before returning error
            with latest_request_lock:
                if request_id != latest_request_id:
                     print(f"Request {request_id} cancelled during error handling by newer request {latest_request_id}.")
                     return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200
            return jsonify({'error': str(e)}), 500
        # The 'processing_lock' is automatically released when exiting the 'with processing_lock:' block

@app.route('/cancel_processing', methods=['POST'])
def cancel_processing():
    """Endpoint to cancel any ongoing processing by marking the current latest request as invalid"""
    global latest_request_id
    with latest_request_lock:
        # Setting latest_request_id to None ensures that if a thread is currently
        # processing or waiting for the lock, its request_id check will fail,
        # causing it to cancel itself.
        latest_request_id = None
    print("Cancellation requested. latest_request_id set to None.")
    return jsonify({'status': 'success', 'message': 'Cancellation signal sent'})


if __name__ == '__main__':
    # Note: debug=True enables the Flask reloader and debugger, which can sometimes
    # interact in complex ways with threading. For production, use a production
    # WSGI server (like Gunicorn or uWSGI) configured with multiple worker processes
    # and/or threads. The lock mechanism implemented here will work correctly
    # within a multi-threaded Flask application served by a production server.
    app.run(host='0.0.0.0', port=5000, debug=True)
