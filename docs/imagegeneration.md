# Image Generation

EasilyAI supports AI-powered image generation through OpenAI's DALL-E models. This guide covers everything from basic image creation to advanced techniques for getting the best results.

## Basic Image Generation

### Simple Image Creation

Generate images with a text prompt:

```python
from easilyai import create_app

# Create an app with DALL-E
app = create_app("ImageGenerator", "openai", "your-openai-key", "dall-e-3")

# Generate an image
image_url = app.request(
    "A serene mountain landscape at sunset",
    task_type="generate_image"
)

print(f"Generated image: {image_url}")
```

### Available Models

EasilyAI supports different DALL-E models:

```python
from easilyai import create_app

# DALL-E 3 - Latest model with highest quality
dalle3_app = create_app("DALLE3", "openai", "your-key", "dall-e-3")

# DALL-E 2 - Previous version, more cost-effective
dalle2_app = create_app("DALLE2", "openai", "your-key", "dall-e-2")

prompt = "A futuristic city with flying cars"

# Generate with DALL-E 3
image_v3 = dalle3_app.request(prompt, task_type="generate_image")

# Generate with DALL-E 2
image_v2 = dalle2_app.request(prompt, task_type="generate_image")

print(f"DALL-E 3: {image_v3}")
print(f"DALL-E 2: {image_v2}")
```

## Image Parameters

### Size Options

Control the dimensions of generated images:

```python
from easilyai import create_app

app = create_app("ImageGen", "openai", "your-key", "dall-e-3")

prompt = "A beautiful garden with colorful flowers"

# Different size options
sizes = ["1024x1024", "1792x1024", "1024x1792"]

for size in sizes:
    image_url = app.request(
        prompt,
        task_type="generate_image",
        size=size
    )
    print(f"Size {size}: {image_url}")
```

### Quality Settings

Adjust image quality (DALL-E 3 only):

```python
from easilyai import create_app

app = create_app("ImageGen", "openai", "your-key", "dall-e-3")

prompt = "A detailed portrait of a wise old wizard"

# Standard quality (faster, less expensive)
standard_image = app.request(
    prompt,
    task_type="generate_image",
    quality="standard"
)

# HD quality (slower, more expensive, higher detail)
hd_image = app.request(
    prompt,
    task_type="generate_image",
    quality="hd"
)

print(f"Standard: {standard_image}")
print(f"HD: {hd_image}")
```

### Style Control

Influence the artistic style:

```python
from easilyai import create_app

app = create_app("ImageGen", "openai", "your-key", "dall-e-3")

# Natural style (more photo-realistic)
natural_image = app.request(
    "A cat sitting on a windowsill",
    task_type="generate_image",
    style="natural"
)

# Vivid style (more artistic and vibrant)
vivid_image = app.request(
    "A cat sitting on a windowsill",
    task_type="generate_image",
    style="vivid"
)

print(f"Natural: {natural_image}")
print(f"Vivid: {vivid_image}")
```

## Advanced Image Generation

### Detailed Prompts

Create more specific and detailed prompts for better results:

```python
from easilyai import create_app

app = create_app("DetailedGen", "openai", "your-key", "dall-e-3")

# Basic prompt
basic_prompt = "A house"

# Detailed prompt
detailed_prompt = """
A cozy two-story cottage with a thatched roof, surrounded by a wildflower garden. 
The house has warm yellow walls, blue shutters, and a wooden front door. 
Smoke is gently rising from the chimney. The scene is set during golden hour 
with soft, warm lighting. Style: watercolor illustration.
"""

basic_image = app.request(basic_prompt, task_type="generate_image")
detailed_image = app.request(detailed_prompt, task_type="generate_image")

print(f"Basic: {basic_image}")
print(f"Detailed: {detailed_image}")
```

### Artistic Styles

Specify artistic styles in your prompts:

```python
from easilyai import create_app

app = create_app("ArtisticGen", "openai", "your-key", "dall-e-3")

base_prompt = "A majestic lion in the savanna"

styles = [
    "photorealistic",
    "oil painting",
    "watercolor",
    "digital art",
    "pencil sketch",
    "impressionist style",
    "anime style",
    "minimalist design"
]

for style in styles:
    styled_prompt = f"{base_prompt}, {style}"
    image_url = app.request(styled_prompt, task_type="generate_image")
    print(f"{style}: {image_url}")
```

### Composition and Perspective

Control composition and camera angles:

```python
from easilyai import create_app

app = create_app("CompositionGen", "openai", "your-key", "dall-e-3")

subject = "A red sports car"

compositions = [
    "close-up shot",
    "wide angle view",
    "bird's eye view",
    "low angle shot",
    "profile view",
    "three-quarter view"
]

for composition in compositions:
    prompt = f"{subject}, {composition}, professional photography"
    image_url = app.request(prompt, task_type="generate_image")
    print(f"{composition}: {image_url}")
```

## Multiple Images

### Generating Multiple Variations

Create multiple images from the same prompt:

```python
from easilyai import create_app

app = create_app("MultiGen", "openai", "your-key", "dall-e-2")  # DALL-E 2 supports multiple images

prompt = "A peaceful zen garden with a small pond"

# Generate multiple variations (DALL-E 2 only)
images = []
for i in range(3):
    image_url = app.request(
        prompt,
        task_type="generate_image",
        n=1  # Number of images per request
    )
    images.append(image_url)

for i, image in enumerate(images, 1):
    print(f"Variation {i}: {image}")
```

### Batch Image Generation

Generate multiple different images:

```python
from easilyai import create_app
import time

def batch_generate_images(prompts, app):
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            image_url = app.request(prompt, task_type="generate_image")
            results.append({"prompt": prompt, "image": image_url})
            print(f"Generated {i+1}/{len(prompts)}: {prompt}")
            
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            results.append({"prompt": prompt, "error": str(e)})
    
    return results

# Usage
app = create_app("BatchGen", "openai", "your-key", "dall-e-3")

prompts = [
    "A mystical forest with glowing mushrooms",
    "A futuristic space station orbiting Earth",
    "A vintage library with towering bookshelves",
    "A serene beach at sunrise with palm trees"
]

results = batch_generate_images(prompts, app)

for result in results:
    if "image" in result:
        print(f"✓ {result['prompt']}: {result['image']}")
    else:
        print(f"✗ {result['prompt']}: {result['error']}")
```

## Working with Generated Images

### Downloading Images

Download and save generated images:

```python
from easilyai import create_app
import requests
import os

app = create_app("ImageDownloader", "openai", "your-key", "dall-e-3")

# Generate image
prompt = "A majestic eagle soaring over mountains"
image_url = app.request(prompt, task_type="generate_image")

# Download and save
def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
    else:
        print(f"Failed to download image: {response.status_code}")

# Create directory if it doesn't exist
os.makedirs("generated_images", exist_ok=True)

# Download image
filename = "generated_images/eagle_mountains.png"
download_image(image_url, filename)
```

### Image Variations

Create variations of existing images (DALL-E 2 only):

```python
from easilyai import create_app

app = create_app("ImageVariations", "openai", "your-key", "dall-e-2")

# Note: This requires the OpenAI client directly for image uploads
# EasilyAI focuses on text-to-image generation
# For image variations, you would need to use the OpenAI client directly
```

## Practical Examples

### Logo Generation

Create logos for brands:

```python
from easilyai import create_app

app = create_app("LogoGen", "openai", "your-key", "dall-e-3")

# Logo prompts
logo_prompts = [
    "A minimalist logo for a coffee shop called 'Bean There', incorporating a coffee bean, warm colors, vector style",
    "A modern logo for a tech startup called 'CloudSync', featuring cloud imagery, blue and white colors, clean design",
    "A playful logo for a pet store called 'Pawsome Pets', with paw prints, bright colors, friendly design"
]

for prompt in logo_prompts:
    logo_url = app.request(prompt, task_type="generate_image")
    print(f"Logo: {logo_url}")
```

### Product Visualization

Generate product images for e-commerce:

```python
from easilyai import create_app

app = create_app("ProductViz", "openai", "your-key", "dall-e-3")

products = [
    "A sleek wireless bluetooth headphone in matte black, product photography, white background",
    "A vintage leather messenger bag, brown color, professional product shot, soft lighting",
    "A modern ceramic coffee mug with geometric patterns, white and blue, clean background"
]

for product in products:
    image_url = app.request(product, task_type="generate_image")
    print(f"Product image: {image_url}")
```

### Concept Art

Create concept art for creative projects:

```python
from easilyai import create_app

app = create_app("ConceptArt", "openai", "your-key", "dall-e-3")

# Game concept art
game_concepts = [
    "A mystical floating island with ancient ruins, waterfalls cascading into clouds below, fantasy art style",
    "A cyberpunk city street at night with neon signs, rain-slicked pavement, dramatic lighting",
    "A steampunk airship with brass fittings and steam pipes, flying through cloudy skies"
]

for concept in game_concepts:
    artwork_url = app.request(concept, task_type="generate_image")
    print(f"Concept art: {artwork_url}")
```

## Best Practices

### 1. Effective Prompting

```python
# Be specific and descriptive
good_prompt = "A golden retriever puppy playing in a sunny meadow filled with wildflowers, professional pet photography, shallow depth of field"

# Avoid vague prompts
bad_prompt = "A dog"

# Include style and mood
styled_prompt = "A cozy cabin in the woods during autumn, warm lighting, painted in the style of Bob Ross"
```

### 2. Use Technical Terms

```python
# Photography terms
photo_prompt = "A portrait of a woman, 85mm lens, soft lighting, bokeh background, professional headshot"

# Art terms
art_prompt = "A landscape painting, impressionist style, visible brushstrokes, vibrant colors, plein air technique"
```

### 3. Specify Quality and Resolution

```python
from easilyai import create_app

app = create_app("QualityGen", "openai", "your-key", "dall-e-3")

# For high-quality images
hq_prompt = "A detailed architectural rendering of a modern house, 4K resolution, professional visualization"

image_url = app.request(
    hq_prompt,
    task_type="generate_image",
    quality="hd",
    size="1024x1024"
)
```

### 4. Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException

def safe_image_generation(prompt, max_retries=3):
    app = create_app("SafeGen", "openai", "your-key", "dall-e-3")
    
    for attempt in range(max_retries):
        try:
            image_url = app.request(prompt, task_type="generate_image")
            return image_url
        
        except EasilyAIException as e:
            if "content policy" in str(e).lower():
                print(f"Content policy violation: {e}")
                return None
            elif attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                continue
            else:
                print(f"All attempts failed: {e}")
                return None
    
    return None

# Usage
result = safe_image_generation("A peaceful garden scene")
if result:
    print(f"Success: {result}")
else:
    print("Failed to generate image")
```

## Limitations and Considerations

### Content Policy

- Images must comply with OpenAI's content policy
- Avoid generating images of real people without consent
- No violent, sexual, or harmful content

### Technical Limits

- DALL-E 3: 1 image per request
- DALL-E 2: Up to 10 images per request
- Size limitations based on model
- Rate limits apply

### Cost Considerations

- DALL-E 3: More expensive but higher quality
- DALL-E 2: More cost-effective for multiple variations
- HD quality costs more than standard

## Troubleshooting

### Common Issues

1. **Content Policy Violations**: Revise prompts to comply with guidelines
2. **Rate Limiting**: Implement delays between requests
3. **Low Quality Results**: Use more specific prompts and DALL-E 3
4. **Download Failures**: Handle HTTP errors when downloading images

### Example Error Handling

```python
from easilyai import create_app
from easilyai.exceptions import EasilyAIException
import time

def robust_image_generation(prompt):
    app = create_app("RobustGen", "openai", "your-key", "dall-e-3")
    
    try:
        image_url = app.request(prompt, task_type="generate_image")
        return image_url
    
    except EasilyAIException as e:
        if "rate limit" in str(e).lower():
            print("Rate limit reached. Waiting 60 seconds...")
            time.sleep(60)
            return robust_image_generation(prompt)  # Retry
        else:
            print(f"Error: {e}")
            return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

Image generation opens up creative possibilities for content creation, product visualization, and artistic projects. Next, explore [Text-to-Speech](/texttospeech) or learn about [Pipelines](/pipelines) to combine image generation with other AI tasks.