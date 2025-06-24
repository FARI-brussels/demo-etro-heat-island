# Heatmap POC

This project generates a heatmap from an input image, considering different weather scenarios.

## Project Setup

### Prerequisites
- Python 3.x
- pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd heatmapPOC
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies using `pyproject.toml`:**
    This project uses a `pyproject.toml` file to manage dependencies with Poetry.
    If you don't have Poetry installed, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

    Once Poetry is installed, navigate to the `backend` directory and run:
    ```bash
    cd backend
    poetry install
    ```
    This command will read the `pyproject.toml` file, resolve the dependencies, and install them into your virtual environment.

4.  **Ensure `random_forest.joblib` is present:**
    The application requires a pre-trained model file named `random_forest.joblib` to be present in the `backend` directory. Make sure you have this file.

### Running the Backend Server

Once the setup is complete, you can run the Flask backend server:

```bash
cd backend
python app.py
```

The server will start on `http://0.0.0.0:5000`.

## API Documentation

The backend exposes the following API endpoints:

### 1. Process Image

*   **Endpoint:** `/process_image`
*   **Method:** `POST`
*   **Description:** Receives an image, processes it to detect ArUco markers, rectifies the image based on these markers, and then generates a heat matrix and related data based on the specified mode.
*   **Request Body:** JSON object
    ```json
    {
        "image": "<base64_encoded_image_string>",
        "mode": "<string>"
    }
    ```
    *   `image`: Base64 encoded string of the JPEG or PNG image.
    *   `mode`: Specifies the weather scenario. Possible values:
        *   `"summer_day"`: Simulates a hot summer day.
        *   `"summer_night"`: Simulates a summer night (potentially rainy).
        *   `"real_time"`: Fetches current weather data from an external API.

*   **Success Response (200 OK):**
    ```json
    {
        "status": "success",
        "source_image": "<base64_encoded_rectified_jpeg_image_string>",
        "heat_matrix": [[<float>, ...], ...],
        "temperature": <float>,
        "weather_data": {
            "alt": <integer>,
            "short_wave": <float>,
            "t2m": <float>,         // Temperature in Celsius
            "rel_humid": <float>,   // Relative humidity in %
            "wind_speed": <float>,  // Wind speed in m/s
            "max_t2m": <float>,     // Max daily temperature in Celsius
            "min_t2m": <float>,     // Min daily temperature in Celsius
            "KERNEL_250M_PX": <integer>,
            "surrounding": <integer>,
            "weather": {
                "id": <integer>,
                "main": "<string>",         // e.g., "Clear", "Clouds", "Rain"
                "description": "<string>",  // e.g., "clear sky", "few clouds"
                "icon": "<string>"          // Weather icon ID
            }
        }
    }
    ```
    *   `source_image`: Base64 encoded string of the rectified input image (JPEG format).
    *   `heat_matrix`: A 2D array (list of lists) representing the temperature values for each cell in the processed grid.
    *   `temperature`: The average temperature (score) calculated from the `heat_matrix`.
    *   `weather_data`: A dictionary containing the weather parameters used for the heatmap calculation. The structure of `weather` sub-dictionary matches the OpenWeatherMap format.

*   **Error Responses:**
    *   `400 Bad Request`:
        *   `{"error": "No image data provided"}`
        *   `{"error": "Could not decode image data"}`
        *   `{"status": "error", "message": "Could not find 4 ArUco markers or rectify"}`
    *   `500 Internal Server Error`:
        *   `{"error": "<description_of_server_error>"}`
    *   `200 OK (with status cancelled)`:
        *   `{'status': 'cancelled', 'message': 'Request superseded by newer request'}`: This occurs if a new `/process_image` request is received before the current one finishes processing. The older request is cancelled.

### 2. Cancel Processing

*   **Endpoint:** `/cancel_processing`
*   **Method:** `POST`
*   **Description:** Signals the server to cancel any ongoing image processing for the `/process_image` endpoint. This is useful if a new image needs to be processed and the previous request is no longer relevant.
*   **Request Body:** None
*   **Success Response (200 OK):**
    ```json
    {
        "status": "success",
        "message": "Cancellation signal sent"
    }
    ```

## How it Works

1.  **Image Submission:** The client sends a base64 encoded image and a processing `mode` to the `/process_image` endpoint.
2.  **Request Handling:**
    *   The server uses a locking mechanism (`latest_request_id`, `latest_request_lock`, `processing_lock`) to ensure that only the most recent request is processed. If a new request comes in while an old one is still in the queue or being processed, the older one is marked for cancellation.
3.  **Image Preprocessing:**
    *   The received image is decoded and rotated 180 degrees.
    *   **ArUco Marker Detection:** The system attempts to find four ArUco markers in the image.
    *   **Rectification:** If four markers are found, the area defined by the inner corners of these markers is extracted and perspective-transformed into a square image (300x300 pixels by default). If markers are not found, an error is returned.
4.  **Heatmap Generation (`create_heatmap.py`):**
    *   The rectified image is passed to the `create_heatmap` function along with the selected `mode`.
    *   **Mode Selection & Weather Data:**
        *   `summer_day` / `summer_night`: Uses predefined weather parameters.
        *   `real_time`: Fetches current weather data from the Brussels Mobility Twin API (requires a valid API key configured in `create_heatmap.py`).
    *   **Image to Matrix:** The RGB image is converted into a grid matrix where each cell represents a land use type (Water, Green, Impervious) based on HSV color analysis.
    *   **Kernel Application:** Convolutional kernels (approximating circular areas of 250m and 1km) are applied to this matrix to calculate the percentage of each land use type in the surrounding area of each cell.
    *   **Temperature Prediction:** A pre-trained Random Forest model (`random_forest.joblib`) predicts the temperature for each cell based on its land use composition and the weather parameters.
    *   **Output:** The function returns the raw heat matrix (temperature values), an overall temperature score (average), and the weather data used.
5.  **Response:** The backend sends back the rectified source image (base64 encoded), the heat matrix, the average temperature, and the weather data.

### Initialization (`initialize_resources` in `create_heatmap.py`)

*   Called once when the Flask application starts.
*   Loads the `random_forest.joblib` model.
*   Pre-calculates the convolution kernels based on parameters in `create_heatmap.py`.
*   Fetches initial real-time weather data to have `EXTRA_PARAMETERS` populated. If this fails, the server will still start but `/process_image` calls with `real_time` mode might fail until the API is reachable. 