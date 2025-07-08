import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "EasilyAI",
  description: "A python library that simplifies the usage of AI!",
  base: "/EasilyAI/",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Get Started', link: '/overview' },
      { text: 'Guide', link: '/guide' },
      { text: 'API Reference', link: '/api' },
      { text: 'Examples', link: '/examples' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Overview', link: '/overview' },
          { text: 'Installation', link: '/installation' },
          { text: 'Quick Start', link: '/quickstart' },
          { text: 'Basic Examples', link: '/examples' },
        ]
      },
      {
        text: 'User Guide',
        items: [
          { text: 'Creating Apps', link: '/appcreation' },
          { text: 'Working with Services', link: '/services' },
          { text: 'Text Generation', link: '/textgeneration' },
          { text: 'Image Generation', link: '/imagegeneration' },
          { text: 'Text to Speech', link: '/texttospeech' },
          { text: 'Pipelines', link: '/pipelines' },
          { text: 'Error Handling', link: '/errorhandling' },
        ]
      },
      {
        text: 'Advanced Features',
        items: [
          { text: 'Custom AI Services', link: '/customai' },
          { text: 'Configuration', link: '/configuration' },
          { text: 'Performance Tips', link: '/performance' },
        ]
      },
      {
        text: 'AI Services',
        items: [
          { text: 'OpenAI', link: '/openai' },
          { text: 'Anthropic (Claude)', link: '/anthropic' },
          { text: 'Google Gemini', link: '/gemini' },
          { text: 'X.AI Grok', link: '/grok' },
          { text: 'Ollama', link: '/ollama' },
          { text: 'Hugging Face', link: '/huggingface' },
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Core Classes', link: '/api' },
          { text: 'Service Classes', link: '/api/services' },
          { text: 'Pipeline System', link: '/api/pipelines' },
          { text: 'Custom AI Framework', link: '/api/customai' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/GustyCube/EasilyAI' },
      { icon: 'python', link: 'https://pypi.org/project/EasilyAI/'}
    ]
  }
})
