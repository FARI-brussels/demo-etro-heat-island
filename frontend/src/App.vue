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
                <span>{{ maxTemp.toFixed(1) }}°C</span>
                <span>{{ minTemp.toFixed(1) }}°C</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <p v-if="temperature !== null" class="temperature-display">Average Temperature: {{ temperature.toFixed(1) }} °C</p>
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
const backendUrl = 'http://localhost:5000/process_image';

const generateHeatmap = (normalizedMatrix, minVal, maxVal) => {
  // Create a canvas element
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Set canvas size to match matrix dimensions
  canvas.width = normalizedMatrix[0].length;
  canvas.height = normalizedMatrix.length;
  
  // Create image data
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const data = imageData.data;
  
  // Apply plasma colormap to normalized values
  for (let i = 0; i < normalizedMatrix.length; i++) {
    for (let j = 0; j < normalizedMatrix[i].length; j++) {
      const value = normalizedMatrix[i][j];
      const idx = (i * canvas.width + j) * 4;
      
      // Plasma colormap approximation
      const r = Math.min(255, Math.max(0, value * 1.5));
      const g = Math.min(255, Math.max(0, value * 0.8));
      const b = Math.min(255, Math.max(0, value * 0.5));
      
      data[idx] = r;     // R
      data[idx + 1] = g; // G
      data[idx + 2] = b; // B
      data[idx + 3] = 255; // A
    }
  }
  
  // Put the image data on the canvas
  ctx.putImageData(imageData, 0, 0);
  
  // Convert canvas to data URL
  return canvas.toDataURL('image/png');
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
    
    // Generate heatmap from normalized matrix
    const normalizedMatrix = data.normalized_matrix;
    minTemp.value = data.min_temp;
    maxTemp.value = data.max_temp;
    heatmapImageUrl.value = generateHeatmap(normalizedMatrix, minTemp.value, maxTemp.value);
    
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
  width: 40%;
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
}

.legend {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

.legend-gradient {
  width: 100%;
  height: 20px;
  background: linear-gradient(to right, 
    rgb(0, 0, 255),    /* Cold */
    rgb(255, 0, 255),  /* Medium */
    rgb(255, 255, 0)   /* Hot */
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
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
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
</style>
