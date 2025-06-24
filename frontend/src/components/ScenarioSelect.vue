<template>
  <div class="settings">
    <FDropdown
      v-model="settingsOpen"
      location="bottom-left"
      icon="settings"
      class="bg-color-blue mt-l"
      on-dark
      small
    >
      <div class="weather-menu rounded-s">
        <div
          v-for="{ value, label, icon } in scenarios"
          :key="value"
          class="scenario-item rounded-s p-xs"
          :class="{ selected: value === scenario }"
          @click="() => selectScenario(value)"
        >
          <!-- <component :is="icon" /> -->
          <span class="font-weight-black font-size-body"> {{ label[locale] }} </span>
        </div>

      </div>
    </FDropdown>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { FDropdown } from 'fari-component-library'
import IconSummer from './icons/IconSummer.vue'
import IconWinter from './icons/IconWinter.vue'
import IconAutumn from './icons/IconAutumn.vue'

type Scenarios = 'summer_day' | 'summer_night' | 'real_time'


defineProps<{
  locale: string
  scenario: Scenarios
}>()

const emit = defineEmits<{
  (e: 'scenario', value: Scenarios): void
}>()

const settingsOpen = ref(false)

function selectScenario(value: Scenarios) {
  emit('scenario', value)
  settingsOpen.value = false
}

const scenarios = [
  {
    label: {
      en: 'Summer day',
      'fr-FR': 'Journée d’été',
      nl: 'Zomerse dag'
    },
    value: 'summer_day',
    icon: IconSummer
  },
  {
    label: {
      en: 'Summer night',
      'fr-FR': "Nuit d’été",
      nl: 'Zomernacht'
    },
    value: 'summer_night',
    icon: IconAutumn
  },
  {
    label: {
      en: 'Real-time',
      'fr-FR': "Temps réel",
      nl: 'Real-time'
    },
    value: 'real_time',
    icon: IconWinter
  },
] as const
</script>

<style scoped lang="scss">
.settings {
  margin-left: 2rem;
}

.scenario-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  background-color: #2f519c;
  transition: all 200ms ease-in-out;
}

.selected {
  background-color: #4393de90;
}

.agent-selector {
  margin-right: auto;
}

:deep() {
  .dropdown-menu {
    width: fit-content;
    height: fit-content;
    .content {
      width: 400px;
      padding: 1rem;
      transition: 400ms ease-in-out;
      overflow: hidden;
      display: flex;
    }
  }
}

.weather-menu {
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  width: 100%;
}
</style>
