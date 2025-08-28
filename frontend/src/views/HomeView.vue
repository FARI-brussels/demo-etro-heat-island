<template>
  <div class="view bg-color-blue">
    <FButtonIcon
      name="tooltip"
      class="tooltip"
      color="blue-light"
      small
      @click="toggleInfoCard"
    />
    <div class="heading">
      <div class="title"> <FTitle class="title"> HEATMAP BRUSSELS </FTitle> </div>
      </div>
    <FButton
      label="Start demo"
      type="primary"
      class="start-button"
      on-dark
      @click="router.push('/interactive')"
    />

    <img v-if="data?.image" :src="data.image" alt="Background" class="background-image rounded" />
<FSlideTransition :show="showInfoCard">
      <FCard
        v-if="showInfoCard"
        @close="toggleInfoCard"
        @update:locale="setLocale"
        class="card"
      >
        {{ data.description[locale] }}

        <div class="researchers-container">
          <span class="researchers">
            research head:
            <span class="research-head color-black">
              {{ data.research_head }}
            </span>
          </span>
          <span class="researchers">
            research lead:
            <span class="research-lead color-black">
              {{ data.research_lead }}
            </span>
          </span>
          <div class="flex">
            <img
              v-for="sdg in data.sdg_images"
              :key="sdg"
              :src="sdg"
              class="sdg mr-md mt-sm"
            />
          </div>
        </div>
        <template #footer>
          <div v-if="data?.logos">
            <img
              v-for="logo in data.logos"
              :src="logo"
              :key="logo"
              class="card-logo"
            />
          </div>
        </template>
      </FCard>
    </FSlideTransition>
    <div class="backdrop" :class="{ 'backdrop-active': showInfoCard }"></div>
    <FFooter class="footer" />
  </div>

</template>


<script setup lang="ts"> 
import {
  FTitle,
  FFooter,
  FButton,
  FButtonIcon,
  FSlideTransition,
  FCard,
} from 'fari-component-library'
import { useRouter } from 'vue-router'
import { useDataStore } from '@/stores/cms'
import { ref, onMounted } from 'vue'
import { storeToRefs } from 'pinia'


const showInfoCard = ref(false)
const toggleInfoCard = () => (showInfoCard.value = !showInfoCard.value)

const { data, locale } = storeToRefs(useDataStore())
const { getData, setLocale } = useDataStore()

onMounted(getData)

const router = useRouter()

</script>

<style scoped lang="scss">
.view {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  height: 100%;
  width: 100%;
  position: relative;
}

.tooltip {
  z-index: 2;
  position: absolute;
  top: 7rem;
  right: 32rem;
}
.card {
  position: absolute;
  top: 20%;
  left: 20%;
  z-index: 3;
}

.background-image {
  width: 50vw;
  position: absolute;
  top: 25%;
  left: 25%;
  z-index: 1;
}

.researchers-container {
  margin-top: auto;
  display: flex;
  flex-direction: column;
}

.researchers {
  color: #888;
  text-transform: uppercase;
}

.research-head,
.research-lead {
  text-transform: none;
}

.card-logo {
  height: 3.5rem;
  margin-right: 2rem;
}

.sdg {
  height: 3.5rem;
  width: 3.5rem;
}



.heading {
  width: 100%;
  position: absolute;
  top: 8rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space;

  .title {
    gap: 1rem;
    display: flex;
    flex-direction: column;
  }
}

.start-button {
  display: inline-block;
  width: auto;
  position: absolute;
  top: 50%;
  left: 45%;
  z-index: 2;
}

.footer {
  margin-top: auto;
  margin-bottom: 1.8rem;
  position: absolute;
  bottom: 0;
  width: 100%;
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

  &-active {
    visibility: visible;
    opacity: 1;
    background-color: rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(2px);
    transition: all 300ms;
  }
}

</style>