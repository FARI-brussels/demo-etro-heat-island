# Webcam ArUco Processor

This project consists of a Vue.js frontend and a Flask backend. The frontend captures webcam images, detects ArUco markers and changes, and sends the images to the backend for processing. The backend processes the images and sends back the processed images to be displayed on the frontend.

## Project Structure

```
.
├── frontend/           # Vue.js frontend
│   ├── src/            # Vue source code
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── backend/            # Flask backend
    ├── app.py          # Flask application
    └── requirements.txt  # Python dependencies
```

## Prerequisites

- Node.js (v14+)
- npm or yarn
- Python (3.8+)
- pip

## Setup and Running

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn
   ```

3. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

   This will start the Vue.js development server, typically at http://localhost:5173.

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

   This will start the Flask server at http://localhost:5000.

## Usage

1. Open the frontend URL in your browser (e.g., http://localhost:5173).
2. Click the "Start Capture" button to activate your webcam.
3. When significant changes are detected in the webcam image and ArUco markers are found, the image will be sent to the backend.
4. The backend will process the image and return the processed image, which will be displayed on the right side of the frontend.

## ArUco Markers

To test this application, you'll need to print ArUco markers. You can find printable ArUco markers online, or generate them using OpenCV.

For this application, we're using the 6x6_250 ArUco dictionary.

## Notes

- The current implementation has placeholders for ArUco marker detection and image change detection.
- The backend processes images by applying edge detection and adding heatmap-like effects around detected ArUco markers.
- For development purposes, CORS is enabled on the backend to allow requests from the frontend. 