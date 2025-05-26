<template>
  <div id="app-container">
    <WebcamDisplay @image-captured="handleImageCapture" />
    <div class="processed-images-container">
      <p v-if="!sourceImageUrl && !heatmapImageUrl">Waiting for processed images...</p>
      
      <div v-else class="images-wrapper">
        <div class="image-box">
          <h3>Source Image</h3>
          <img v-if="sourceImageUrl" :src="sourceImageUrl" alt="Source Image"/>
        </div>
        
        <div class="image-box">
          <h3>Heatmap</h3>
          <div class="heatmap-container">
            <img v-if="heatmapImageUrl" :src="heatmapImageUrl" alt="Heatmap Image"/>
            <div class="legend">
              <div class="legend-gradient"></div>
              <div class="legend-labels">
                <span>{{ minTemp.toFixed(1) }}°C</span>
                <span>{{ maxTemp.toFixed(1) }}°C</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <p v-if="temperature !== null" class="temperature-display">Average Temperature: {{ temperature.toFixed(1) }} °C</p>

      <div v-if="weatherData" class="weather-info-container">
        <h3>Current Weather</h3>
        <div class="weather-item">
          <span>Temperature:</span>
          <span>{{ weatherData.t2m.toFixed(1) }} °C</span>
        </div>
        <div class="weather-item">
          <span>Humidity:</span>
          <span>{{ weatherData.rel_humid.toFixed(0) }} %</span>
        </div>
        <div class="weather-item">
          <span>Wind Speed:</span>
          <span>{{ weatherData.wind_speed.toFixed(1) }} m/s</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import WebcamDisplay from './components/WebcamDisplay.vue';

const sourceImageUrl = ref(null);
const heatmapImageUrl = ref(null);
const temperature = ref(null);
const minTemp = ref(null);
const maxTemp = ref(null);
const weatherData = ref(null);
const backendUrl = 'http://localhost:5000/process_image';

const generateHeatmap = (heatMatrix, minVal, maxVal) => {
  // Create a canvas element
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Get the source image dimensions
  const sourceImg = document.querySelector('.image-box img');
  if (!sourceImg) {
    console.error('Source image not found');
    return null;
  }
  
  // Set canvas size to match source image dimensions
  canvas.width = sourceImg.naturalWidth;
  canvas.height = sourceImg.naturalHeight;
  
  // Create image data
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const data = imageData.data;
  
  // Calculate scaling factors
  const scaleX = canvas.width / heatMatrix[0].length;
  const scaleY = canvas.height / heatMatrix.length;
  
  // Normalize the heat matrix to 0-255 range
  const normalizedMatrix = heatMatrix.map(row => 
    row.map(value => {
      if (maxVal === minVal) return 0;
      return Math.floor(255 * (value - minVal) / (maxVal - minVal));
    })
  );
  
  // Apply plasma colormap to normalized values with scaling
  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      // Map canvas coordinates to matrix coordinates
      const matrixX = Math.floor(x / scaleX);
      const matrixY = Math.floor(y / scaleY);
      
      // Get value from normalized matrix
      const value = normalizedMatrix[matrixY][matrixX];
      const idx = (y * canvas.width + x) * 4;
      
      // Improved plasma colormap approximation
      // Cold (blue) -> Medium (purple) -> Hot (yellow)
      let r, g, b;
      if (value < 85) {
        // Blue to Purple transition
        r = Math.floor(value * 3);
        g = 0;
        b = 255;
      } else if (value < 170) {
        // Purple to Yellow transition
        r = 255;
        g = Math.floor((value - 85) * 3);
        b = Math.floor(255 - (value - 85) * 3);
      } else {
        // Yellow to Red transition
        r = 255;
        g = Math.floor(255 - (value - 170) * 3);
        b = 0;
      }
      
      data[idx] = r;     // R
      data[idx + 1] = g; // G
      data[idx + 2] = b; // B
      data[idx + 3] = 255; // A
    }
  }
  
  // Put the image data on the canvas
  ctx.putImageData(imageData, 0, 0);
  
  // Create a new canvas for the rotated and flipped image
  const rotatedCanvas = document.createElement('canvas');
  const rotatedCtx = rotatedCanvas.getContext('2d');
  
  // Set the rotated canvas size (swapped width and height for 90-degree rotation)
  rotatedCanvas.width = canvas.height;
  rotatedCanvas.height = canvas.width;
  
  // Save the current context state
  rotatedCtx.save();
  
  // Translate to the center of the canvas
  rotatedCtx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
  
  // Rotate 90 degrees clockwise
  rotatedCtx.rotate(-Math.PI / 2);
  
  // Flip horizontally
  rotatedCtx.scale(-1, 1);
  
  // Draw the original canvas centered
  rotatedCtx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
  
  // Restore the context state
  rotatedCtx.restore();
  
  // Convert rotated canvas to data URL
  return rotatedCanvas.toDataURL('image/png');
};

const handleImageCapture = async (imageDataUrl) => {
  try {
    console.log("Image captured, sending to backend...");
    
    // Extract base64 data from the data URL
    const base64Data = imageDataUrl.split(',')[1];
    
    // Send the image to the backend
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: base64Data })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    // Process the response
    const data = await response.json();
    
    // Update source image URL
    sourceImageUrl.value = `data:image/jpeg;base64,${data.source_image}`;
    
    // Generate heatmap from heat matrix
    const heatMatrix = data.heat_matrix;
    weatherData.value = data.weather_data;
    minTemp.value = Math.min(...heatMatrix.flat());
    maxTemp.value = Math.max(...heatMatrix.flat());
    heatmapImageUrl.value = generateHeatmap(heatMatrix, minTemp.value, maxTemp.value);
    
    // Update temperature
    temperature.value = data.temperature;
    console.log("Received processed images from backend");
  } catch (error) {
    console.error("Error sending image to backend:", error);
  }
};
</script>

<style scoped>
#app-container {
  display: flex;
  justify-content: space-around;
  align-items: flex-start;
  width: 100%;
  padding: 20px;
}

.processed-images-container {
  width: 45%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.images-wrapper {
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 20px;
}

.image-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 45%;
}

.image-box h3 {
  margin: 5px 0;
  color: #333;
}

.heatmap-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

.legend {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-width: 500px;
}

.legend-gradient {
  width: 100%;
  height: 20px;
  background: linear-gradient(to right, 
    rgb(0, 0, 255),     /* Cold (Blue) */
    rgb(255, 0, 255),   /* Medium (Purple) */
    rgb(255, 255, 0),   /* Hot (Yellow) */
    rgb(255, 0, 0)      /* Very Hot (Red) */
  );
  border-radius: 4px;
}

.legend-labels {
  display: flex;
  justify-content: space-between;
  width: 100%;
  margin-top: 5px;
  font-size: 12px;
  color: #666;
}

img {
  width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
  object-fit: contain;
}

.temperature-display {
  margin-top: 20px;
  font-size: 18px;
  font-weight: bold;
  color: #e63946;
  background-color: #f1faee;
  padding: 8px 15px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.weather-info-container {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
  width: 100%;
  max-width: 500px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.weather-info-container h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #333;
  font-size: 16px;
  text-align: center;
}

.weather-item {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
  color: #555;
  padding: 5px 0;
}

.weather-item span:first-child {
  font-weight: bold;
}
</style>
