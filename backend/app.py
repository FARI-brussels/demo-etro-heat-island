from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import cv2 as cv
from create_heatmap import create_heatmap
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the base64 encoded image from the request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert to OpenCV format if needed (BGR)
        if len(image_np.shape) > 2 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) > 2 and image_np.shape[2] == 3:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process the image - just detect ArUco markers
        processed_image = crop_and_rectify_aruco_square(image_np)
        result_img, score = create_heatmap(processed_image)
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(result_img)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        processed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'processed_image': processed_base64
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

def detect_aruco_markers(image):
    """
    Detect ArUco markers in the image.
    If at least 4 markers are found, highlight them in the image.
    """
    # Make a copy of the image to avoid modifying the original
    output = image.copy()

    # Convert to grayscale for marker detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    # Using DICT_4X4_50 as in the user's provided function
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    # Add text to indicate how many markers were found
    marker_count = 0 if ids is None else len(ids)
    cv2.putText(
        output,
        f"ArUco markers detected: {marker_count}/4",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if marker_count >= 4 else (0, 0, 255),
        2
    )

    # If markers are detected, draw them on the image
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

    # Return the output image and the detection results
    return output, corners, ids, rejected

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
    output_image, corners, ids, rejected = detect_aruco_markers(image.copy()) # Use a copy to avoid modifying the original input

    # Check if exactly 4 markers are detected
    if ids is None or len(ids) != 4:
        # Draw error message on the output_image from detect_aruco_markers
        cv2.putText(
            output_image,
            f"Error: Exactly 4 ArUco markers required. Found: {0 if ids is None else len(ids)}",
            (10, 70), # Position below marker count
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, # Slightly smaller font
            (0, 0, 255),
            2
        )
        print(f"Error: Exactly 4 ArUco markers required. Found: {0 if ids is None else len(ids)}")
        return output_image # Return the image with the error message

    try:
        # Calculate the center of each marker
        marker_centers = []
        for i, corner in enumerate(corners):
            # Each corner is an array of 4 points (the corners of the marker)
            # Calculate the center as the mean of these points
            center = np.mean(corner.squeeze(), axis=0)
            marker_centers.append((center, i))  # Store the center with the marker index
            
            # Draw the center of the marker for visualization
            cv2.circle(output_image, tuple(map(int, center)), 5, (255, 0, 0), -1)
            
            # Add marker ID next to center
            cv2.putText(
                output_image,
                f"ID:{ids[i][0]}",
                (int(center[0]) + 10, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
        
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
            error_msg = f"Could not find a marker in each quadrant: TL:{top_left}, TR:{top_right}, BR:{bottom_right}, BL:{bottom_left}"
            print(error_msg)
            cv2.putText(
                output_image,
                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            return output_image
        
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
        
        # Draw the selected corners on the output image for visualization
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        for i, corner in enumerate(quad_corners):
            cv2.circle(output_image, tuple(map(int, corner)), 10, corner_colors[i], -1)
            # Add a label
            cv2.putText(
                output_image,
                f"Corner {i}",
                (int(corner[0]) + 15, int(corner[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                corner_colors[i],
                2
            )
        
        # Draw lines connecting the selected corners
        for i in range(4):
            pt1 = tuple(map(int, quad_corners[i]))
            pt2 = tuple(map(int, quad_corners[(i + 1) % 4]))
            cv2.line(output_image, pt1, pt2, (0, 255, 255), 2)
        
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
        error_msg = f"Error during rectification: {str(e)}"
        print(error_msg)
        cv2.putText(
            output_image,
            error_msg[:50] + "..." if len(error_msg) > 50 else error_msg,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        return output_image
    
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 