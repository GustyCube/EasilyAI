import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "EasilyAI",
  description: "A unified Python library for seamless AI integration across multiple providers",
  base: "/EasilyAI/",
  head: [
    ['link', { rel: 'icon', href: '/EasilyAI/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],
  
  markdown: {
    lineNumbers: true,
    container: {
      tipLabel: '💡 Tip',
      warningLabel: '⚠️ Warning',
      dangerLabel: '❌ Danger',
      infoLabel: 'ℹ️ Info',
      detailsLabel: '📋 Details'
    }
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/logo.svg',
    
    search: {
      provider: 'local'
    },
    
    editLink: {
      pattern: 'https://github.com/GustyCube/EasilyAI/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },
    
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 EasilyAI Contributors'
    },
    
    lastUpdated: {
      text: 'Updated at',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },
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
      { 
        icon: {
          svg: '<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>PyPI</title><path d="M6.4 19.2H8l.7-2.4h1.9l.7 2.4h1.6l-2.5-8H8.9l-2.5 8zm3.1-6.8l.6 2.2H9.5l.6-2.2h-.6zm5.5 6.8h1.5v-3.2h1.8c1.6 0 2.5-.9 2.5-2.4s-.9-2.4-2.5-2.4H15v8zm1.5-6.8v1.6h1.3c.6 0 1-.3 1-.8s-.4-.8-1-.8h-1.3zM6.4 4.8H8l.7 2.4h1.9l.7-2.4h1.6l-2.5 8H8.9l-2.5-8zm3.1 6.8l.6-2.2H9.5l.6 2.2h-.6zm5.5-6.8h1.5v3.2h1.8c1.6 0 2.5.9 2.5 2.4s-.9 2.4-2.5 2.4H15v-8zm1.5 6.8v-1.6h1.3c.6 0 1 .3 1 .8s-.4.8-1 .8h-1.3z"/></svg>'
        }, 
        link: 'https://pypi.org/project/EasilyAI/'
      }
    ]
  }
})
