<template>
  <div id="app-container">
    <WebcamDisplay @image-captured="handleImageCapture" />
    <div class="processed-image-container">
      <img v-if="processedImageUrl" :src="processedImageUrl" alt="Processed Image" class="rotated-image" />
      <p v-else>Waiting for processed image...</p>
      <p v-if="temperature !== null" class="temperature-display">Temperature: {{ temperature }} °C</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import WebcamDisplay from './components/WebcamDisplay.vue';

const processedImageUrl = ref(null);
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
    
    // Update the processed image URL only on success
    processedImageUrl.value = `data:image/jpeg;base64,${data.processed_image}`;
    // Update temperature if it exists in the response
    temperature.value = data.temperature || null;
    console.log("Received processed image from backend");
  } catch (error) {
    console.error("Error sending image to backend:", error);
    // No longer replacing with placeholder image on error
    // Keep the last successful image by not updating processedImageUrl
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

.processed-image-container {
  width: 45%; /* Adjust as needed */
  display: flex;
  flex-direction: column;
  align-items: center;
}

img {
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
}

.rotated-image {
  transform: rotate(180deg);
}

.temperature-display {
  margin-top: 10px;
  font-size: 18px;
  font-weight: bold;
}
</style>
