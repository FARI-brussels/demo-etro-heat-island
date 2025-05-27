from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import time
import threading
# Import initialize_resources and the updated create_heatmap
from create_heatmap import create_heatmap, initialize_resources

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Variable to store the ID of the latest request received
latest_request_id = None
# Lock to protect access to the latest_request_id variable
latest_request_lock = threading.Lock()

# Lock to ensure only one thread executes the heavy image processing at a time
processing_lock = threading.Lock()

# --- Helper functions (ArUco detection remains the same) ---

def detect_aruco_markers(image_rgb):
    """
    Detect ArUco markers in the RGB image.
    If at least 4 markers are found, highlight them in the image.
    """
    # Convert RGB to grayscale for marker detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids, rejected

def order_points(pts):
    # Sort points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype="float32")

def crop_and_rectify_aruco_square(image_rgb, target_size=(300, 300)):
    """
    Detects 4 ArUco markers in an RGB image and rectifies the inner square.
    Args:
        image_rgb: The input RGB image (NumPy array).
        target_size: The desired size (width, height) of the output square image.
    Returns:
        The rectified square RGB image (NumPy array) or None.
    """
    corners, ids, rejected = detect_aruco_markers(image_rgb) # Expects RGB
    if ids is None or len(ids) != 4:
        return None
    try:
        marker_centers = []
        for i, corner_set in enumerate(corners):
            center = np.mean(corner_set.squeeze(), axis=0)
            marker_centers.append((center, i))

        centroid = np.mean([mc[0] for mc in marker_centers], axis=0)
        top_left, top_right, bottom_right, bottom_left = None, None, None, None

        for center, idx in marker_centers:
            if center[0] < centroid[0] and center[1] < centroid[1]: top_left = idx
            elif center[0] > centroid[0] and center[1] < centroid[1]: top_right = idx
            elif center[0] > centroid[0] and center[1] > centroid[1]: bottom_right = idx
            elif center[0] < centroid[0] and center[1] > centroid[1]: bottom_left = idx

        if None in [top_left, top_right, bottom_right, bottom_left]: return None

        quad_corners = np.zeros((4, 2), dtype=np.float32)
        quad_corners[0] = corners[top_left].squeeze()[2]    # TL marker -> BR corner
        quad_corners[1] = corners[top_right].squeeze()[3]   # TR marker -> BL corner
        quad_corners[2] = corners[bottom_right].squeeze()[0] # BR marker -> TL corner
        quad_corners[3] = corners[bottom_left].squeeze()[1]  # BL marker -> TR corner
        
        (width, height) = target_size
        dst_pts = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(quad_corners, dst_pts)
        rectified_image_rgb = cv2.warpPerspective(image_rgb, M, target_size) # Operates on RGB
        return rectified_image_rgb
    except Exception as e:
        print(f"Error in crop_and_rectify_aruco_square: {e}")
        return None

# --- Flask Routes ---

@app.route('/process_image', methods=['POST'])
def process_image():
    global latest_request_id
    request_id = str(time.time())

    with latest_request_lock:
        latest_request_id = request_id
        print(f"Registered request: {request_id}. Latest is now: {latest_request_id}")

    with processing_lock:
        with latest_request_lock:
            if request_id != latest_request_id:
                print(f"Request {request_id} is not the latest ({latest_request_id}). Cancelling.")
                return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200

        print(f"Proceeding with processing for request: {request_id}")
        try:
            data = request.json
            if not data or 'image' not in data:
                with latest_request_lock: # Check before returning error
                    if request_id != latest_request_id: return jsonify({'status': 'cancelled', 'message': 'Request superseded'}), 200
                return jsonify({'error': 'No image data provided'}), 400

            image_data_base64 = data['image']
            mode = data['mode']
            image_data_decoded = base64.b64decode(image_data_base64)
            
            # Decode directly to NumPy array (BGR format by default with cv2.imdecode)
            image_np_rgb = cv2.imdecode(np.frombuffer(image_data_decoded, np.uint8), cv2.IMREAD_COLOR)
            if image_np_rgb is None:
                with latest_request_lock: # Check before returning error
                    if request_id != latest_request_id: return jsonify({'status': 'cancelled', 'message': 'Request superseded'}), 200
                return jsonify({'error': 'Could not decode image data'}), 400

            # Rotate image 180 degrees using OpenCV
            image_np_rgb_rotated = cv2.rotate(image_np_rgb , cv2.ROTATE_180)
            
            # Process the image - ArUco detection (expects RGB)
            rectified_rgb_image = crop_and_rectify_aruco_square(image_np_rgb_rotated)
            if rectified_rgb_image is None:
                with latest_request_lock: # Check before returning error
                    if request_id != latest_request_id: return jsonify({'status': 'cancelled', 'message': 'Request superseded'}), 200
                return jsonify({'status': 'error', 'message': 'Could not find 4 ArUco markers or rectify'}), 400

            with latest_request_lock: # Check before heavy heatmap step
                if request_id != latest_request_id:
                     print(f"Request {request_id} cancelled before heatmap by newer request {latest_request_id}.")
                     return jsonify({'status': 'cancelled', 'message': 'Request superseded by newer request'}), 200

            # Generate normalized matrix and get temperature values
            heat_matrix, score, weather_data = create_heatmap(rectified_rgb_image, mode)

            # Convert RGB to BGR for PIL Image then to base64 JPEG
            rectified_bgr_image = cv2.cvtColor(rectified_rgb_image, cv2.COLOR_RGB2BGR)
            pil_src_image = Image.fromarray(rectified_bgr_image)
            
            src_buffer = io.BytesIO()
            pil_src_image.save(src_buffer, format='JPEG', quality=85)
            src_base64 = base64.b64encode(src_buffer.getvalue()).decode('utf-8')

            print(f"Successfully processed request {request_id} and sending response.")
            return jsonify({
                'status': 'success',
                'source_image': src_base64,
                'heat_matrix': heat_matrix.tolist(),
                'temperature': score,
                'weather_data': weather_data
            })

        except Exception as e:
            print(f"Error processing image for request {request_id}: {e}")
            raise
            # Check if still latest before returning error
            with latest_request_lock:
                if request_id != latest_request_id: return jsonify({'status': 'cancelled', 'message': 'Request superseded'}), 200
            return jsonify({'error': str(e)}), 500

@app.route('/cancel_processing', methods=['POST'])
def cancel_processing():
    global latest_request_id
    with latest_request_lock:
        latest_request_id = None
    print("Cancellation requested. latest_request_id set to None.")
    return jsonify({'status': 'success', 'message': 'Cancellation signal sent'})


if __name__ == '__main__':
    print("Starting Flask app...")
    # Initialize resources once at startup
    try:
        initialize_resources()
        print("Application resources initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize application resources: {e}")
        # Depending on the severity, you might want to exit or handle this
        # For now, we'll let Flask try to start, but it will likely fail on requests.

    # When debug=True, Flask's reloader runs the main module twice.
    # To prevent initialize_resources() from running twice (and for other reasons too in prod),
    # set use_reloader=False if you need debug=True for other features.
    # For production, a proper WSGI server like Gunicorn or uWSGI should be used.
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
