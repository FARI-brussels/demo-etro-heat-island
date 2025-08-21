<template>
  <div class="webcam-container rounded">
    <div class="video-wrapper">
      <FButton v-if="showWebcam" label="Capture image" @click="captureImage" class="capture-button"/>
      <FButton v-else label="Toggle webcam" @click="toggleWebcam" class="toggle-button"/>
      <h3 v-if="showWebcam" class="color-white">Webcam</h3>
      
      <video
        v-show="showWebcam"
        ref="videoPlayer"
        autoplay
        playsinline
        @loadedmetadata="onVideoLoaded"
        class="rotated-video rounded"
      />
      <FButtonIcon v-if="showWebcam" @click="toggleWebcam" name="cross" class="cancel-button" color="red" small />
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
import { FAppBar, FButton, FButtonIcon } from 'fari-component-library';

const emit = defineEmits(['image-captured']);
defineProps<{ processedImage?: string }>();


const videoPlayer = ref(null);
const canvasElement = ref(null);
const stream = ref(null);
const status = ref('Initializing...');
const showWebcam = ref(true);
let lastImageData = null;
const videoLoaded = ref(false);


let noChangeStartTime = ref(null);

const onVideoLoaded = () => {
  if (videoPlayer.value && canvasElement.value) {
    canvasElement.value.width = videoPlayer.value.videoWidth;
    canvasElement.value.height = videoPlayer.value.videoHeight;
    videoLoaded.value = true;
    status.value = 'Video loaded. Ready to capture.';
  }
};


const toggleWebcam = () => 
  showWebcam.value = !showWebcam.value;


  const leImg = ref<string | undefined>(undefined)

  function captureImage() {
  if (!videoPlayer.value || !canvasElement.value) return;

  const canvas = canvasElement.value as HTMLCanvasElement;
  const ctx = canvas.getContext('2d');

  if (!ctx) return;

  // Draw current video frame onto canvas
  ctx.drawImage(videoPlayer.value as HTMLVideoElement, 0, 0, canvas.width, canvas.height);

  // Convert canvas to image
  const imageDataUrl = canvas.toDataURL('image/jpeg', 0.7);

  leImg.value = imageDataUrl;
  console.log(imageDataUrl);

  toggleWebcam();
  emit('image-captured', imageDataUrl);
}

async function startWebcam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      status.value = 'Initializing webcam...';
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.value = mediaStream;
      if (videoPlayer.value) videoPlayer.value.srcObject = mediaStream;
      status.value = 'Webcam active, waiting for video to load...';
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
  videoLoaded.value = false;
  lastImageData = null;
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

.capture-button, .toggle-button {
  position: absolute;
  bottom: -4rem;
  left: 30%;
  z-index: 1000;
}

.cancel-button {
  position: absolute;
  top: 4.8rem;
  right: .3rem;
  z-index: 1000;
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