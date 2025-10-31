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
        <FButton label="toggle map" @click="showMap = !showMap"/>
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
    
    <div
      @close="toggleInfoCard"
      @update:locale="setLocale"
      class="card pa-lg bg-color-primary rounded cesium-wrapper" 
      :class="{['cesium-wrapper-visible']: showMap}"
    > 
      <FButtonIcon @click="showMap = !showMap" name="cross" class="close-map-button" color="red" small />
      <div  ref="cesiumContainer" class="viewer-container rounded" />
    </div>

    <div class="backdrop" :class="{ 'backdrop-active': showMap }" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { FAppBar, FButtonIcon , FButton, FSlideTransition, FContainer} from 'fari-component-library';
import ScenarioSelect from '../components/ScenarioSelect.vue';
import WebcamDisplay from '../components/WebcamDisplay.vue';
import WeatherItem from '../components/WeatherItem.vue';
import HeatmapDisplay from '../components/HeatmapDisplay.vue';
import { generateHeatmap } from '../utils/heatmap';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

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

const showMap = ref(false);
const cesiumContainer = ref<HTMLDivElement | null>(null);
let viewer: Cesium.Viewer | null = null;

const center = ref([4.3517, 50.8503]);
const zoom = ref(10);
const bbox = computed(() => {
  const zoomLevel = zoom.value;
  const lon = center.value[0];
  const lat = center.value[1];
  const delta = 0.05 / (2 ** (zoomLevel - 11));
  return [lon - delta, lat - delta, lon + delta, lat + delta];
});

const initializeViewer = async () => {
  if (!cesiumContainer.value) {
    console.error('Cesium container not found');
    return;
  }

  // Set Cesium Ion access token
  Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJhY2E3ZDhlNC03Yjc0LTQzM2QtYmI5My0zYWQ3NjIwOTk0OTciLCJpZCI6Mjc4NzM4LCJpYXQiOjE3NDA0ODg1MjB9.VsZjL6pbKSwR_SBbxUq-KRweOU_P3R8DKjSpeD0EICY';

  try {
    viewer = new Cesium.Viewer(cesiumContainer.value, {
      sceneMode: Cesium.SceneMode.SCENE3D,
      baseLayerPicker: false,
      timeline: false,
      animation: false,
      geocoder: true,
      homeButton: true,
      sceneModePicker: true,
      navigationHelpButton: true,
      infoBox: true,
      selectionIndicator: true,
    });

    console.info('Cesium viewer initialized');

    // Set digital terrain
    viewer.scene.setTerrain(
      new Cesium.Terrain(
        Cesium.CesiumTerrainProvider.fromIonAssetId(3340034),
      ),
    );

    viewer.imageryLayers.removeAll();
    viewer.imageryLayers.addImageryProvider(
      new Cesium.OpenStreetMapImageryProvider({
        url: 'https://tile.openstreetmap.org/',
      })
    );

    viewer.imageryLayers.addImageryProvider(
      new Cesium.WebMapServiceImageryProvider({
        url: 'https://ows.environnement.brussels/air',
        layers: 'urban_heat_islands',
        parameters: {
          service: 'WMS',
          transparent: true,
          format: 'image/png',
          version: '1.1.1',
          styles: 'défaut',
        },
      })
    );

    viewer.camera.setView({
      destination: Cesium.Rectangle.fromDegrees(bbox.value[0], bbox.value[1], bbox.value[2], bbox.value[3]),
    });

    console.log('WMS layer added with bbox:', bbox.value);

    // Load tilesets
    const tilesetUrls = [
      'https://digitaltwin.s3.gra.io.cloud.ovh.net/tileset_manager/2025-08-18_12-30-52/tileset.json',
      'https://digitaltwin.s3.gra.io.cloud.ovh.net/tileset_manager/2025-08-18_12-24-12/tiles/tileset.json',
      'https://digitaltwin.s3.gra.io.cloud.ovh.net/tileset_manager/2025-08-18_12-27-59/tiles/tileset.json'
    ];

    try {
      const promises = tilesetUrls.map(url => Cesium.Cesium3DTileset.fromUrl(url));
      const loadedTilesets = await Promise.all(promises);

      loadedTilesets.forEach((tileset, index) => {
        viewer!.scene.primitives.add(tileset);
        console.log(`Tileset ${index + 1} loaded and added to scene`);
      });

      if (loadedTilesets.length > 0) {
        await viewer!.zoomTo(loadedTilesets[0]);
        console.log('Zoomed to first tileset');
      }
    } catch (tilesetError) {
      console.error('Failed to load tilesets:', tilesetError);
    }
  } catch (error) {
    console.error('Failed to initialize Cesium viewer:', error);
  }
};

onMounted(() => {
  initializeViewer();
});

onBeforeUnmount(() => {
  if (viewer) {
    viewer.destroy();
    viewer = null;
    console.log('Cesium viewer destroyed');
  }
});


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
:deep(.app-bar.appbar) {
  z-index: 2;
}

.cesium-wrapper {
  position: absolute;
  top: -100%; 
  transition: top 0.3s ease-in-out;
  z-Index: 2005;
  /* width: 80vw;  */
  width: 65vw; 
  height: 80vh;
  display: flex;
  padding: 2rem;

    justify-content: center;
    align-items: center;
}

.cesium-wrapper-visible {
    top: 10%; 
  }

  .viewer-container {
    width: 100%;
    height: 100%;
  }

  .close-map-button {
    position: absolute;
    top: -1rem;
    right: -1rem;

  }


  .backdrop {
  visibility: hidden;
  opacity: 0;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  backdrop-filter: blur(0);
  z-index: 2;
  transition: all 100ms;

}

.backdrop-active {
  visibility: visible;
  opacity: 1;
  background-color: rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(2px);
  transition: all 300ms;
}


:deep(.cesium-widget) {
  border-radius: 2rem;
}

:deep(.cesium-viewer-fullscreenContainer) {
  bottom: 1rem;
  right: 1rem;
}

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
  margin-top: 26px;
  font-size: 1.3rem;
  font-weight: bold;
  padding: 8px 15px;
  position: absolute;
  bottom: 4rem;
  right: 20rem;
}

</style>