<template>
  <div class="webcam-container rounded">
    <div class="video-wrapper">
      <h3 v-if="showWebcam" class="color-white">Webcam</h3>
      <video
        v-show="showWebcam"
        ref="videoPlayer"
        autoplay
        playsinline
        @loadedmetadata="onVideoLoaded"
        class="rotated-video rounded"
      />

      <div v-if="!showWebcam && processedImage" class="processed-image-container">
        <h3 class="color-white">Processed Image</h3>
        <img
          :src="processedImage"
          alt="Processed Image"
          class="processed-image rounded"
        />
      </div>
    </div>
    <canvas ref="canvasElement" style="display: none;"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';

const emit = defineEmits(['image-captured']);
const props = defineProps<{ processedImage?: string }>();


const videoPlayer = ref(null);
const canvasElement = ref(null);
const stream = ref(null);
const status = ref('Initializing...');
const isCapturing = ref(false);
const showWebcam = ref(true);
let processInterval = null;
let lastImageData = null;
const videoLoaded = ref(false);
const lastCaptureTime = ref(0);
const CAPTURE_COOLDOWN = 1000;
const FRAME_PROCESS_INTERVAL = 500;
const NO_CHANGE_THRESHOLD = 1000; 

let noChangeStartTime = ref(null);

const onVideoLoaded = () => {
  if (videoPlayer.value && canvasElement.value) {
    canvasElement.value.width = videoPlayer.value.videoWidth;
    canvasElement.value.height = videoPlayer.value.videoHeight;
    videoLoaded.value = true;
    status.value = 'Video loaded. Ready to capture.';

    if (isCapturing.value) startProcessing();
  }
};

const startProcessing = () => {
  if (processInterval) clearInterval(processInterval);
  processInterval = setInterval(processFrame, FRAME_PROCESS_INTERVAL);
};

const processFrame = () => {
  if (!videoPlayer.value || !canvasElement.value || !isCapturing.value) return;

  if (!videoLoaded.value || videoPlayer.value.videoWidth === 0 || videoPlayer.value.videoHeight === 0) 
    return;
  

  const context = canvasElement.value.getContext('2d');
  if (!context) return;

  canvasElement.value.width = videoPlayer.value.videoWidth;
  canvasElement.value.height = videoPlayer.value.videoHeight;
  context.drawImage(videoPlayer.value, 0, 0, canvasElement.value.width, canvasElement.value.height);

  try {
    const currentImageData = context.getImageData(0, 0, canvasElement.value.width, canvasElement.value.height);
    const changeDetected = detectChange(currentImageData.data);

    const now = Date.now();
    if (changeDetected) {

      noChangeStartTime.value = null;
      showWebcam.value = true;
      if (now - lastCaptureTime.value >= CAPTURE_COOLDOWN) {
        const imageDataUrl = canvasElement.value.toDataURL('image/jpeg', 0.7);
        emit('image-captured', imageDataUrl);
        lastImageData = currentImageData.data.slice();
        lastCaptureTime.value = now;
      }
    } else {
      if (!noChangeStartTime.value) noChangeStartTime.value = now; 
      
      if (props.processedImage && now - noChangeStartTime.value >= NO_CHANGE_THRESHOLD) 
        showWebcam.value = false;
      
    }
    if (!lastImageData) lastImageData = currentImageData.data.slice();
    
  } catch (error) {
    console.error('Error processing frame:', error);
    status.value = `Error processing frame: ${error.message}`;
  }
};

const detectChange = (currentFrameData) => {
  if (!lastImageData || lastImageData.length !== currentFrameData.length) return true; 
  

  let diffPixels = 0;
  const threshold = 30;
  const pixelChangeThreshold = 0.07;
  const sampleRate = 4;

  for (let i = 0; i < currentFrameData.length; i += 4 * sampleRate) {
    const r1 = lastImageData[i];
    const g1 = lastImageData[i + 1];
    const b1 = lastImageData[i + 2];
    const r2 = currentFrameData[i];
    const g2 = currentFrameData[i + 1];
    const b2 = currentFrameData[i + 2];

    const diff = Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2);
    if (diff > threshold) diffPixels++;
  }

  const totalPixelsSampled = currentFrameData.length / (4 * sampleRate);
  const changedPercentage = diffPixels / totalPixelsSampled;
  return changedPercentage > pixelChangeThreshold;
};

async function startWebcam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      status.value = 'Initializing webcam...';
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.value = mediaStream;
      if (videoPlayer.value) videoPlayer.value.srcObject = mediaStream;
      status.value = 'Webcam active, waiting for video to load...';
      isCapturing.value = true;
      if (videoLoaded.value) startProcessing();
    } catch (error) {
      console.error('Error accessing webcam:', error);
      status.value = `Error accessing webcam: ${error.message}`;
    }
  } else {
    status.value = 'getUserMedia not supported';
    console.error(status.value);
  }
}

function stopWebcam() {
  if (stream.value) stream.value.getTracks().forEach(track => track.stop());
  if (processInterval) clearInterval(processInterval);
  isCapturing.value = false;
  videoLoaded.value = false;
  lastImageData = null;
  lastCaptureTime.value = 0;
  noChangeStartTime.value = null;
  showWebcam.value = true;
  status.value = 'Webcam stopped.';
}

onMounted(startWebcam);
onBeforeUnmount(stopWebcam);
</script>

<style scoped>
.webcam-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  border-radius: 14px;
  width: 800px;
  height: 800px;
  margin-top: 2rem;
  background-color: #2f519c;
}

.video-wrapper {
  position: relative;
  width: 100%;
  max-width: 640px;
}

video,
.processed-image {
  width: 640px;
  height: auto;
  border: 1px solid #ccc;
  border-radius: 14px;
}

.rotated-video {
  transform: rotate(180deg);
}

.processed-image-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.processed-image-label {
  margin-top: 8px;
  font-size: 14px;
  color: #333;
  text-align: center;
}
</style>