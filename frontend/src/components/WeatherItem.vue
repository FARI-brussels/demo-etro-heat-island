<template>
  <div class="weather-item">
    <span class="label color-white mx-sm">{{ props.label }}</span>
    <Component v-if="iconComponent" :is="iconComponent"  />
    <span  class="value color-white">{{  props.value }}</span>
  </div>
</template>

<script setup lang="ts">
import IconSunny from './icons/IconSunny.vue'
import IconCloudy from './icons/IconCloudy.vue'
import IconRainy from './icons/IconRainy.vue'
import { computed } from 'vue'
import { FIcon } from 'fari-component-library'

const props = defineProps<{
  label: string;
  value: string;
}>();

const iconComponent = computed(() => {
  if(props.label !== 'Condition:') return null
  if(props.value === 'Clouds') return IconCloudy
  if(props.value === 'Rain') return IconRainy
  if(props.value === 'Sun') return IconSunny
}) 

</script>

<style scoped>
.weather-item {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
  color: #555;
  padding: 5px 0;
  text-align: left;
}

.weather-item .label {
  font-weight: bold;
}
</style>