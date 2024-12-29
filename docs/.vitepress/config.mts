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
      { text: 'Installation', link: '/installation' }
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
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/GustyCube/EasilyAI' }
    ]
  }
})
