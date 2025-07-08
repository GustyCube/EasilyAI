<template>
  <div class="feature-card" :class="{ [`feature-card--${variant}`]: variant }">
    <div class="feature-card__icon" v-if="icon">
      <span v-html="icon"></span>
    </div>
    <div class="feature-card__content">
      <h3 class="feature-card__title" v-if="title">{{ title }}</h3>
      <p class="feature-card__description" v-if="description">{{ description }}</p>
      <div class="feature-card__body">
        <slot />
      </div>
    </div>
    <div class="feature-card__badge" v-if="badge">
      <span class="badge" :class="`badge-${badgeType}`">{{ badge }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  title?: string
  description?: string
  icon?: string
  badge?: string
  badgeType?: 'success' | 'warning' | 'danger' | 'info'
  variant?: 'default' | 'highlight' | 'subtle'
}

withDefaults(defineProps<Props>(), {
  badgeType: 'info',
  variant: 'default'
})
</script>

<style scoped>
.feature-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(62, 175, 124, 0.15);
}

.feature-card--highlight {
  background: linear-gradient(135deg, var(--vp-c-brand-soft) 0%, var(--vp-c-brand-soft) 100%);
  border-color: var(--vp-c-brand-1);
}

.feature-card--subtle {
  background: var(--vp-c-default-soft);
  border-color: var(--vp-c-default-1);
}

.feature-card__icon {
  font-size: 2rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 4rem;
  height: 4rem;
  background: var(--vp-c-brand-soft);
  border-radius: 50%;
  margin: 0 auto 1rem auto;
}

.feature-card__content {
  flex: 1;
  text-align: center;
}

.feature-card__title {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
  font-size: 1.25rem;
  font-weight: 600;
}

.feature-card__description {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  line-height: 1.5;
}

.feature-card__body {
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.feature-card__badge {
  position: absolute;
  top: 1rem;
  right: 1rem;
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 500;
  line-height: 1;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: 0.375rem;
}

.badge-success {
  color: #fff;
  background-color: var(--vp-c-brand-1);
}

.badge-warning {
  color: #fff;
  background-color: var(--vp-c-warning-1);
}

.badge-danger {
  color: #fff;
  background-color: var(--vp-c-danger-1);
}

.badge-info {
  color: #fff;
  background-color: var(--vp-c-tip-1);
}
</style>