import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "EasyAI ",
  description: "A python library that simplifies the usage of AI!",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Overview', link: '/overview' },
      { text: 'Installation', link: '/installation' },
      { text: 'Services', link: '/services' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Overview', link: '/overview' },
          { text: 'Installation', link: '/installation' },
          { text: 'Error Handling', link: '/errorhandling' },
        ]
      },
      {
        text: 'Guide',
        items: [
          { text: 'App Creation', link: '/appcreation' },
          { text: 'Text To Speech', link: '/texttospeech' },
          { text: 'Pipelines', link: '/pipelines' },
          { text: 'Custom AI', link: '/customai' }
        ]
      },
      {
        text: 'Services',
        link: '/services',
        items: [
          { text: 'OpenAI', link: '/openai' },
          { text: 'Ollama', link: '/ollama' },
          { text: 'Grok', link: '/grok' },
          { text: 'Anthropic (Claude)', link: '/anthropic' },
          { text: 'Hugging Face', link: '/huggingface' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/GustyCube/EasilyAI' },
      { icon: 'python', link: 'https://pypi.org/project/EasilyAI/'}
    ]
  }
})
