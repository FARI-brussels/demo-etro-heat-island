<template>
  <div id="app-container">
    <FAppBar color="primary" class="appbar">
      <template #navigation>
        <FButtonIcon
          name="chevron-left"
          color="blue-light"
          small
          @click="$emit('exit')"
        ></FButtonIcon>
      </template>
      <template #actions>
        <ScenarioSelect
          ref="settings"
          locale="en"
          :scenario="selectedMode"
          @scenario="(e) => selectedMode = e"
        />
      </template>

      <div class="info-panel">
        <div class="weather" v-if="weatherData">
          <div class="weather-item">
            <img src="/temperature.svg" alt="Temperature" class="weather-icon temperature-icon" />
            <span>{{ weatherData.t2m.toFixed(1) }} °C</span>
          </div>
          <div class="weather-item">
            <img src="/humidity.svg" alt="Humidity" class="weather-icon humidity-icon" />
            <span>{{ weatherData.rel_humid.toFixed(0) }} %</span>
          </div>
          <div class="weather-item">
            <img src="/wind.svg" alt="Wind Speed" class="weather-icon wind-icon" />
            <span>{{ weatherData.wind_speed.toFixed(1) }} m/s</span>
          </div>
          <div class="weather-item" v-if="weatherData.weather?.main">
            <img :src="weatherIcon" alt="Weather Condition" class="weather-icon condition-icon" />
          </div>
        </div>
      </div>

    </FAppBar>

    <div class="controls-container rounded">
      <WebcamDisplay @image-captured="handleImageCapture" class="webcam" :processed-image="sourceImageUrl" />
    </div>
    <div class="processed-images-container bg-color-blue-dark rounded-s">
      <p v-if="!heatmapImageUrl">Waiting for processed images...</p>
      <HeatmapDisplay
        v-else 
        :heatmap-image-url="heatmapImageUrl"
        :min-temp="minTemp"
        :max-temp="maxTemp"
        class="heatmap"
      />
    </div>
   <p v-if="temperature !== null" class="temperature-display color-white">
        Average Temperature: {{ temperature.toFixed(1) }} °C
      </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { FAppBar, FButtonIcon } from 'fari-component-library';
import ScenarioSelect from '../components/ScenarioSelect.vue';
import WebcamDisplay from '../components/WebcamDisplay.vue';
import WeatherItem from '../components/WeatherItem.vue';
import HeatmapDisplay from '../components/HeatmapDisplay.vue';
import { generateHeatmap } from '../utils/heatmap';


interface WeatherData {
  t2m: number;
  rel_humid: number;
  wind_speed: number;
  weather?: {
    icon?: string;
    main?: string;
  };
}

interface HeatmapResponse {
  source_image: string;
  heat_matrix: number[][];
  temperature: number;
  weather_data: WeatherData;
}

const sourceImageUrl = ref<string | null>(null);
const heatmapImageUrl = ref<string | null>(null);
const temperature = ref<number | null>(null);
const minTemp = ref<number | null>(null);
const maxTemp = ref<number | null>(null);
const weatherData = ref<WeatherData | null>(null);
const selectedMode = ref<string>('summer_day');

const backendUrl = "http://localhost:5000/process_image"

const weatherItems = computed(() => {
  if (!weatherData.value) return {};
  
  return {
    temperature: `${weatherData.value.t2m.toFixed(1)} °C`,
    humidity: `${weatherData.value.rel_humid.toFixed(0)} %`,
    windSpeed: `${weatherData.value.wind_speed.toFixed(1)} m/s`,
  }

});

const weatherIcon = computed(() => {
  if (!weatherData.value?.weather?.main) return '/cloud.svg'; // Default icon
  const condition = weatherData.value.weather.main.toLowerCase();
  switch (condition) {
    case 'clear':
      return '/sun.svg';
    case 'clouds':
      return '/clouds.svg';
    case 'rain':
    // case 'drizzle':
      return '/rain.svg';
    case 'snow':
      return '/snow.svg';
    // case 'thunderstorm':
    //   return '/thunderstorm.svg';
    // case 'mist':
    // case 'fog':
    // case 'haze':
    //   return '/fog.svg';
    default:
      return '/cloud.svg'; // Fallback
  }
});


const handleImageCapture = async (imageDataUrl: string) => {
  try {
    const base64Data = imageDataUrl.split(',')[1];

    // const mockImageUrl = '/mock5.jpg';
    // const res = await fetch(mockImageUrl, {
    //   method: 'GET',
    // });
    // const blob = await res.blob();

    // const mockBase64 = await new Promise<string>((resolve, reject) => {
    //   const reader = new FileReader();
    //   reader.onloadend = () => resolve(reader.result!.toString().split(',')[1]);
    //   reader.onerror = reject;
    //   reader.readAsDataURL(blob);
    // });

    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: base64Data, 
        mode: selectedMode.value,
      }),
    });

    if (response.status === 400) return console.warn('Backend returned 400, aborting...');
      
    const data: HeatmapResponse = await response.json();

    if (
      !data.heat_matrix ||
      !Array.isArray(data.heat_matrix) ||
      !data.heat_matrix.length ||
      !data.heat_matrix.every(row => Array.isArray(row))
    ) 
      return console.error('Invalid or empty heat_matrix received from backend:', data.heat_matrix);

    sourceImageUrl.value = `data:image/jpeg;base64,${data.source_image}`;

    const sourceImage = new Image();
    sourceImage.src = sourceImageUrl.value;

    await new Promise<void>((resolve) => {
      sourceImage.onload = () => resolve();

      sourceImage.onerror = () => 
        resolve(() => console.error('Failed to load source image for heatmap generation'));
    });

    weatherData.value = data.weather_data;
    temperature.value = data.temperature;
    const heatMatrix = data.heat_matrix;
    const flatMatrix = heatMatrix.flat();

    if (!flatMatrix.length) 
      return console.error('Heat matrix is empty, cannot calculate min/max');
      
    minTemp.value = Math.min(...flatMatrix);
    maxTemp.value = Math.max(...flatMatrix);

    if (isNaN(minTemp.value) || isNaN(maxTemp.value)) 
      return console.error('Invalid minTemp or maxTemp:', { minTemp: minTemp.value, maxTemp: maxTemp.value });
    
    if (!heatmapImageUrl.value) console.error('Failed to generate heatmap');
    
    heatmapImageUrl.value = generateHeatmap(heatMatrix, minTemp.value, maxTemp.value, sourceImage);

    
  } catch (error) {
    console.error('Error sending image to backend:', error);
  }
};
</script>

<style scoped>
.appbar {
  position: absolute;
  top: 0;
  z-index: 4;
  box-sizing: border-box;
  max-width: 100vw;
  height: 120px;
  background-color: #13377790;
}
.weather-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.weather-icon {
  width: 2rem;
  height: 2rem;
}

#app-container {
  display: flex;
  flex-direction: row;
  justify-content: space-evenly;
  align-items: flex-start;
  width: 100%;
  height: 100%;
  gap: 20px;
  margin-top: 120px;
}

.info-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-right: 30rem;
}

.controls-container {
  display: flex;
  flex-direction: column;
  gap: 15px;
  width: auto;
  align-items: flex-start;
}

.weather {
  display: flex;
  gap: 3rem;
}

.weather-item {
  display: flex;
  justify-content: space-between;
  font-size: 1.4rem;
  color: white;
  font-weight: bold;
  text-align: left;
}

.weather-item span:first-child {
  font-weight: bold;
}

.condition-icon {
  width: 4rem;
  height: 4rem;
}

.processed-images-container {
  width: 45%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.temperature-display {
  margin-top: 20px;
  font-size: 1.3rem;
  font-weight: bold;
  padding: 8px 15px;
  position: absolute;
  bottom: 4rem;
  right: 20rem;
}

</style>