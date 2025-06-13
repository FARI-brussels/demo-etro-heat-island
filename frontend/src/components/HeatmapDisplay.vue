<template>
  <div class="image-box">
    <h3 class="color-white">Heatmap</h3>
    <div class="heatmap-container">
      <img
        v-if="heatmapImageUrl"
        :src="heatmapImageUrl"
        alt="Heatmap Image"
        class="heatmap-image"
      />
      
      <p v-else class="error">Failed to generate heatmap</p>

      <div class="legend">
        <div class="legend-gradient"></div>
        <div v-if="minTemp !== null && maxTemp !== null" class="legend-labels">
          <span>{{ minTemp.toFixed(1) }}°C</span>
          <span>{{ maxTemp.toFixed(1) }}°C</span>
        </div>
        <div v-else class="legend-labels">
          <span>N/A</span>
          <span>N/A</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  heatmapImageUrl: string | null;
  minTemp: number | null;
  maxTemp: number | null;
}>();
</script>

<style scoped>
.image-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  border-radius: 14px;
  width: 800px;
  height: 800px;
  margin-top: 2rem;
}

.heatmap-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

.heatmap-image {
  height: auto;
  width: 640px;
  border: 1px solid #ddd;
  border-radius: 8px;
  object-fit: contain;
}

.error {
  color: #e63946;
  font-size: 14px;
  margin: 10px 0;
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
  background: linear-gradient(
    to right,
    rgb(0, 0, 255),
    rgb(255, 0, 255),
    rgb(255, 255, 0),
    rgb(255, 0, 0)
  );
  border-radius: 4px;
}

.legend-labels {
  display: flex;
  justify-content: space-between;
  width: 100%;
  margin-top: 5px;
  font-size: 1rem;
  color: white;
  font-weight: bold;
}
</style>