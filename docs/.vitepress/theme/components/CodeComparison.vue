<template>
  <div class="code-comparison">
    <div class="code-comparison__header" v-if="title">
      <h3>{{ title }}</h3>
      <p v-if="description">{{ description }}</p>
    </div>
    <div class="code-comparison__content">
      <div class="code-comparison__before">
        <div class="code-comparison__label">
          <span class="code-comparison__icon">❌</span>
          {{ beforeLabel }}
        </div>
        <div class="code-comparison__code">
          <slot name="before" />
        </div>
      </div>
      <div class="code-comparison__after">
        <div class="code-comparison__label">
          <span class="code-comparison__icon">✅</span>
          {{ afterLabel }}
        </div>
        <div class="code-comparison__code">
          <slot name="after" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title?: string
  description?: string
  beforeLabel?: string
  afterLabel?: string
}

withDefaults(defineProps<Props>(), {
  beforeLabel: 'Before',
  afterLabel: 'After'
})
</script>

<style scoped>
.code-comparison {
  margin: 2rem 0;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  overflow: hidden;
}

.code-comparison__header {
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-border);
}

.code-comparison__header h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.code-comparison__header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.code-comparison__content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}

@media (max-width: 768px) {
  .code-comparison__content {
    grid-template-columns: 1fr;
  }
}

.code-comparison__before,
.code-comparison__after {
  position: relative;
}

.code-comparison__before {
  border-right: 1px solid var(--vp-c-border);
}

@media (max-width: 768px) {
  .code-comparison__before {
    border-right: none;
    border-bottom: 1px solid var(--vp-c-border);
  }
}

.code-comparison__label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: var(--vp-c-default-soft);
  border-bottom: 1px solid var(--vp-c-border);
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.code-comparison__icon {
  font-size: 1rem;
}

.code-comparison__code {
  padding: 0;
}

.code-comparison__code :deep(div[class*='language-']) {
  margin: 0;
  border-radius: 0;
  border: none;
}
</style>