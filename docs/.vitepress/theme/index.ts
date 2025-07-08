import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// Import custom components
import FeatureCard from './components/FeatureCard.vue'
import CodeComparison from './components/CodeComparison.vue'
import ApiTable from './components/ApiTable.vue'
import InstallationSteps from './components/InstallationSteps.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // https://vitepress.dev/guide/extending-default-theme#layout-slots
    })
  },
  enhanceApp({ app, router, siteData }) {
    // Register global components
    app.component('FeatureCard', FeatureCard)
    app.component('CodeComparison', CodeComparison)
    app.component('ApiTable', ApiTable)
    app.component('InstallationSteps', InstallationSteps)
  }
} satisfies Theme