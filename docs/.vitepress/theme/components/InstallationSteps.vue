<template>
  <div class="installation-steps">
    <div class="installation-steps__header" v-if="title">
      <h3>{{ title }}</h3>
      <p v-if="description">{{ description }}</p>
    </div>
    <div class="installation-steps__list">
      <div 
        v-for="(step, index) in steps" 
        :key="index"
        class="installation-step"
        :class="{ 'installation-step--active': activeStep === index }"
      >
        <div class="installation-step__number">{{ index + 1 }}</div>
        <div class="installation-step__content">
          <h4 class="installation-step__title">{{ step.title }}</h4>
          <p class="installation-step__description" v-if="step.description">
            {{ step.description }}
          </p>
          <div class="installation-step__code" v-if="step.code">
            <div class="installation-step__tabs" v-if="step.tabs">
              <button 
                v-for="tab in step.tabs"
                :key="tab.name"
                class="installation-step__tab"
                :class="{ 'installation-step__tab--active': activeTab[index] === tab.name }"
                @click="setActiveTab(index, tab.name)"
              >
                {{ tab.label }}
              </button>
            </div>
            <div v-if="step.tabs">
              <div 
                v-for="tab in step.tabs"
                :key="tab.name"
                v-show="activeTab[index] === tab.name"
                v-html="tab.code"
              />
            </div>
            <div v-else v-html="step.code" />
          </div>
          <div class="installation-step__note" v-if="step.note">
            <div class="installation-step__note-icon">💡</div>
            <div class="installation-step__note-text">{{ step.note }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'

interface Tab {
  name: string
  label: string
  code: string
}

interface Step {
  title: string
  description?: string
  code?: string
  tabs?: Tab[]
  note?: string
}

interface Props {
  title?: string
  description?: string
  steps: Step[]
  activeStep?: number
}

const props = withDefaults(defineProps<Props>(), {
  activeStep: -1
})

const activeTab = reactive<Record<number, string>>({})

// Initialize active tabs for each step
props.steps.forEach((step, index) => {
  if (step.tabs && step.tabs.length > 0) {
    activeTab[index] = step.tabs[0].name
  }
})

const setActiveTab = (stepIndex: number, tabName: string) => {
  activeTab[stepIndex] = tabName
}
</script>

<style scoped>
.installation-steps {
  margin: 2rem 0;
}

.installation-steps__header {
  margin-bottom: 2rem;
}

.installation-steps__header h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.installation-steps__header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.installation-steps__list {
  position: relative;
}

.installation-steps__list::before {
  content: '';
  position: absolute;
  left: 1.5rem;
  top: 3rem;
  bottom: 3rem;
  width: 2px;
  background: var(--vp-c-border);
  z-index: 1;
}

.installation-step {
  position: relative;
  display: flex;
  gap: 1.5rem;
  margin-bottom: 3rem;
  padding-left: 0.5rem;
}

.installation-step--active .installation-step__number {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
}

.installation-step__number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3rem;
  height: 3rem;
  border: 2px solid var(--vp-c-border);
  border-radius: 50%;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-weight: bold;
  font-size: 1.125rem;
  z-index: 2;
  position: relative;
  flex-shrink: 0;
}

.installation-step__content {
  flex: 1;
  min-width: 0;
}

.installation-step__title {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
  font-size: 1.25rem;
  font-weight: 600;
}

.installation-step__description {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.installation-step__code {
  margin: 1rem 0;
}

.installation-step__tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--vp-c-border);
}

.installation-step__tab {
  background: none;
  border: none;
  padding: 0.5rem 1rem;
  color: var(--vp-c-text-2);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.installation-step__tab:hover {
  color: var(--vp-c-text-1);
}

.installation-step__tab--active {
  color: var(--vp-c-brand-1);
  border-bottom-color: var(--vp-c-brand-1);
}

.installation-step__note {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  margin-top: 1rem;
  padding: 1rem;
  background: var(--vp-c-tip-soft);
  border: 1px solid var(--vp-c-tip-1);
  border-radius: 8px;
}

.installation-step__note-icon {
  font-size: 1.125rem;
  flex-shrink: 0;
}

.installation-step__note-text {
  color: var(--vp-c-text-1);
  line-height: 1.5;
}

.installation-step__code :deep(div[class*='language-']) {
  margin: 0;
  border-radius: 8px;
}

@media (max-width: 768px) {
  .installation-step {
    gap: 1rem;
    padding-left: 0;
  }
  
  .installation-steps__list::before {
    left: 1.25rem;
  }
  
  .installation-step__number {
    width: 2.5rem;
    height: 2.5rem;
    font-size: 1rem;
  }
  
  .installation-step__tabs {
    flex-wrap: wrap;
  }
  
  .installation-step__tab {
    padding: 0.375rem 0.75rem;
    font-size: 0.875rem;
  }
}
</style>