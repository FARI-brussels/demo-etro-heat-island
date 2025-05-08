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
          <img v-if="heatmapImageUrl" :src="heatmapImageUrl" alt="Heatmap Image"/>
        </div>
      </div>
      
      <p v-if="temperature !== null" class="temperature-display">Temperature: {{ temperature }} °C</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import WebcamDisplay from './components/WebcamDisplay.vue';

const sourceImageUrl = ref(null);
const heatmapImageUrl = ref(null);
const temperature = ref(null);
const backendUrl = 'http://localhost:5000/process_image'; // Flask backend URL

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
    
    // Update both image URLs
    sourceImageUrl.value = `data:image/jpeg;base64,${data.source_image}`;
    heatmapImageUrl.value = `data:image/jpeg;base64,${data.heatmap_image}`;
    
    // Update temperature if it exists in the response
    temperature.value = data.temperature || null;
    console.log("Received processed images from backend");
  } catch (error) {
    console.error("Error sending image to backend:", error);
    // No longer replacing with placeholder image on error
    // Keep the last successful images by not updating URLs
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
  width: 45%; /* Adjust as needed */
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
  width: 50%;
}

.image-box h3 {
  margin: 5px 0;
  color: #333;
}

img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.rotated-image {
  transform: rotate(180deg);
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
