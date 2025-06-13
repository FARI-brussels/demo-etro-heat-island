<template>
  <div class="layout">
    <RouterView v-slot="{ Component }">
      <transition name="fade">
        <component :is="Component" />
      </transition>
    </RouterView>
  </div>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<style scoped>
.layout {
  width: 100vw;
  height: 100vh;
  background-color: #183e91;
}

.logo {
  display: block;
  margin: auto 0 13rem 0;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
