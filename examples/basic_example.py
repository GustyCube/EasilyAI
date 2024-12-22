import easilyai
from easilyai.custom_ai import CustomAIService

# Register a Custom AI
class MyCustomAI(CustomAIService):
    def generate_text(self, prompt):
        return f"Custom AI Text: {prompt}"
    
    def text_to_speech(self, text, **kwargs):
        return f"Custom TTS Output: {text}"

easilyai.register_custom_ai("my_custom_ai", MyCustomAI)

# Create OpenAI App
app = easilyai.create_app(
    name="openai_app",
    service="openai",
    apikey="YOUR_OPENAI_API_KEY",
    model="gpt-4"
)

# Run a pipeline
pipeline = easilyai.EasilyAIPipeline(app)
pipeline.add_task("generate_text", "Tell me a story about a talking car.")
pipeline.add_task("generate_image", "A red futuristic talking car with glowing headlights.")
pipeline.add_task("text_to_speech", "Here is a talking car in a futuristic world!")

results = pipeline.run()
for task_result in results:
    print(f"Task: {task_result['task']}\nResult: {task_result['result']}\n")

# Run a TTS-specific app with OpenAI
tts_app = easilyai.create_tts_app(
    name="tts_app",
    service="openai",
    apikey="YOUR_OPENAI_API_KEY",
    model="tts-1"
)

# Generate speech with a specific voice
tts_file = tts_app.request_tts(
    text="Hello, I am your AI assistant!",
    tts_model="tts-1",
    voice="onyx",
    output_file="hello_ai.mp3"
)
print(f"TTS output saved to: {tts_file}")

# Example using Custom AI for TTS
custom_tts_app = easilyai.create_tts_app(
    name="custom_tts",
    service="my_custom_ai",
    model="v1"
)
print(custom_tts_app.request_tts("This is a custom AI TTS example."))
