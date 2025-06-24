<template>
  <div class="view bg-color-blue">
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

    <FFooter class="footer" />
  </div>

</template>


<script setup lang="ts"> 
import {
  FTitle,
  FFooter,
  FButton,

} from 'fari-component-library'

import { useRouter } from 'vue-router'

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

</style>