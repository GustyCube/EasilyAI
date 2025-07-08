<template>
  <div class="api-table">
    <div class="api-table__header" v-if="title">
      <h3>{{ title }}</h3>
      <p v-if="description">{{ description }}</p>
    </div>
    <div class="api-table__container">
      <table class="api-table__table">
        <thead>
          <tr>
            <th v-for="header in headers" :key="header.key" :class="header.class">
              {{ header.label }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in rows" :key="index">
            <td v-for="header in headers" :key="header.key" :class="header.class">
              <div v-if="header.key === 'name' && row.required" class="api-table__name">
                <code>{{ row[header.key] }}</code>
                <span class="api-table__required">*</span>
              </div>
              <code v-else-if="header.key === 'name'" class="api-table__name">
                {{ row[header.key] }}
              </code>
              <code v-else-if="header.key === 'type'" class="api-table__type">
                {{ row[header.key] }}
              </code>
              <div v-else-if="header.key === 'description'" class="api-table__description">
                {{ row[header.key] }}
                <div v-if="row.example" class="api-table__example">
                  <strong>Example:</strong> <code>{{ row.example }}</code>
                </div>
              </div>
              <span v-else>{{ row[header.key] }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Header {
  key: string
  label: string
  class?: string
}

interface Row {
  [key: string]: any
  required?: boolean
  example?: string
}

interface Props {
  title?: string
  description?: string
  headers: Header[]
  rows: Row[]
}

defineProps<Props>()
</script>

<style scoped>
.api-table {
  margin: 2rem 0;
}

.api-table__header {
  margin-bottom: 1rem;
}

.api-table__header h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.api-table__header p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.api-table__container {
  overflow-x: auto;
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
}

.api-table__table {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
  background: var(--vp-c-bg);
}

.api-table__table th {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-text-1);
  font-weight: 600;
  text-align: left;
  padding: 12px 16px;
  border-bottom: 1px solid var(--vp-c-border);
}

.api-table__table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  vertical-align: top;
}

.api-table__table tbody tr:hover {
  background: var(--vp-c-bg-soft);
}

.api-table__name {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
}

.api-table__required {
  color: var(--vp-c-danger-1);
  font-weight: bold;
}

.api-table__type {
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 0.125rem 0.375rem;
  border-radius: 4px;
}

.api-table__description {
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

.api-table__example {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 4px;
  font-size: 0.875rem;
}

.api-table__example code {
  background: transparent;
  padding: 0;
  color: var(--vp-c-text-1);
}

/* Responsive column widths */
.api-table__table th:first-child,
.api-table__table td:first-child {
  width: 20%;
  min-width: 120px;
}

.api-table__table th:nth-child(2),
.api-table__table td:nth-child(2) {
  width: 15%;
  min-width: 100px;
}

.api-table__table th:last-child,
.api-table__table td:last-child {
  width: 65%;
  min-width: 200px;
}

@media (max-width: 768px) {
  .api-table__table {
    font-size: 0.875rem;
  }
  
  .api-table__table th,
  .api-table__table td {
    padding: 8px 12px;
  }
}
</style>