---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "EasilyAI"
  text: "Unified AI Integration"
  tagline: One Python library to rule them all - seamlessly integrate OpenAI, Anthropic, Google Gemini, X.AI Grok, and Ollama with a consistent API
  image:
    src: /hero-image.png
    alt: EasilyAI Hero
  actions:
    - theme: brand
      text: Get Started 🚀
      link: /overview
    - theme: alt
      text: View Examples
      link: /examples
    - theme: alt
      text: API Reference
      link: /api

features:
  - icon: 🎲
    title: Universal Interface
    details: One consistent API for all major AI providers. Switch between OpenAI, Claude, Gemini, Grok, and Ollama effortlessly.
  - icon: ⚡
    title: Lightning Fast Setup
    details: Get running in under 60 seconds. Install, configure, and start generating AI content with just 3 lines of code.
  - icon: 🔧
    title: Highly Extensible
    details: Built-in support for custom AI services. Register your own models and integrate seamlessly with the existing ecosystem.
  - icon: 🎨
    title: Multi-Modal Support
    details: Text generation, image creation, and text-to-speech all in one package. Perfect for comprehensive AI applications.
  - icon: 🚀
    title: Production Ready
    details: Robust error handling, retry logic, and comprehensive logging. Built for enterprise-grade applications.
  - icon: 📊
    title: Pipeline Architecture
    details: Chain multiple AI operations together with our powerful pipeline system for complex workflows.
---

<div class="home-content">

## Quick Start Example

::: code-group

```python [Basic Usage]
from easilyai import create_app

# Create an app with any provider
app = create_app("MyAI", "openai", "your-api-key", "gpt-4")

# Generate text
response = app.request("text", "Write a haiku about coding")
print(response)
```

```python [Image Generation]
from easilyai import create_app

# Use OpenAI for image generation
app = create_app("ImageAI", "openai", "your-api-key", "dall-e-3")

# Generate an image
image_url = app.request("image", "A futuristic robot in a library")
print(f"Generated image: {image_url}")
```

```python [Pipeline Example]
from easilyai import EasilyAIPipeline

# Create a pipeline for complex workflows
pipeline = EasilyAIPipeline()
pipeline.add_task("generate_text", "Write a story about AI")
pipeline.add_task("generate_image", "Illustrate: {previous_output}")

results = pipeline.run()
print(results)
```

:::

## Supported Providers

<div class="provider-grid">
  <div class="provider-card">
    <h3>🤖 OpenAI</h3>
    <p>GPT-4, DALL-E, Whisper</p>
    <span class="badge badge-success">Full Support</span>
  </div>
  <div class="provider-card">
    <h3>🧠 Anthropic</h3>
    <p>Claude 3.5 Sonnet, Haiku</p>
    <span class="badge badge-success">Text Generation</span>
  </div>
  <div class="provider-card">
    <h3>🌐 Google Gemini</h3>
    <p>Gemini Pro, Gemini Vision</p>
    <span class="badge badge-success">Text Generation</span>
  </div>
  <div class="provider-card">
    <h3>✨ X.AI Grok</h3>
    <p>Grok-1, Grok-Vision</p>
    <span class="badge badge-success">Text Generation</span>
  </div>
  <div class="provider-card">
    <h3>💻 Ollama</h3>
    <p>Local LLMs, Llama, Mistral</p>
    <span class="badge badge-success">Text Generation</span>
  </div>
  <div class="provider-card">
    <h3>🤗 Hugging Face</h3>
    <p>Transformers, Diffusers</p>
    <span class="badge badge-warning">Experimental</span>
  </div>
</div>

## Why Choose EasilyAI?

::: tip 💡 Developer Experience
Stop juggling multiple SDKs and APIs. EasilyAI provides a unified interface that feels natural and consistent across all providers.
:::

::: info 🛡️ Production Ready
Built-in error handling, automatic retries, comprehensive logging, and extensive testing make this perfect for production applications.
:::

::: warning ⚡ Performance Focused
Optimized for speed with efficient request handling, connection pooling, and minimal overhead.
:::

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-number">6+</div>
    <div class="stat-label">AI Providers</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">3</div>
    <div class="stat-label">Task Types</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">100%</div>
    <div class="stat-label">Type Safe</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">1</div>
    <div class="stat-label">Unified API</div>
  </div>
</div>

</div>

<style>
.home-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.provider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.provider-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  transition: all 0.3s ease;
}

.provider-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(62, 175, 124, 0.15);
}

.provider-card h3 {
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.provider-card p {
  margin: 0 0 1rem 0;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
}

.stat-card {
  background: linear-gradient(135deg, var(--vp-c-brand-soft) 0%, var(--vp-c-brand-soft) 100%);
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
}

.stat-number {
  font-size: 3rem;
  font-weight: bold;
  color: var(--vp-c-brand-1);
  line-height: 1;
}

.stat-label {
  margin-top: 0.5rem;
  color: var(--vp-c-text-2);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-size: 0.9rem;
}

@media (max-width: 768px) {
  .home-content {
    padding: 1rem;
  }
  
  .provider-grid,
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .stat-number {
    font-size: 2.5rem;
  }
}
</style>
