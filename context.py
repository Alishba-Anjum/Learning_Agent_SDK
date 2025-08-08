import os
import random
from dataclasses import dataclass
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, RunContextWrapper
from agents.run import RunConfig

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key is missing.")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@dataclass
class WeatherContext:
    location: str
    style: str  

@function_tool
async def get_weather(ctx: RunContextWrapper[WeatherContext]) -> str:
    """Fetches weather for the user's location (mock data)."""
    location = ctx.context.location
    fake_weather = random.choice(["Sunny ☀️", "Rainy 🌧️", "Cloudy ☁️", "Stormy ⛈️"])
    return f"The current weather in {location} is {fake_weather}."


def style_instructions(ctx: RunContextWrapper[WeatherContext], agent: Agent[WeatherContext]) -> str:
    style = ctx.context.style
    if style == "formal":
        return "Respond in a very professional and polite tone."
    elif style == "casual":
        return "Respond in a friendly and relaxed way."
    elif style == "funny":
        return "Respond in a humorous, lighthearted way with jokes."
    else:
        return "Just answer normally."


agent = Agent[WeatherContext](
    name="WeatherBot",
    instructions=style_instructions,
    model=model,
    tools=[get_weather]
)


while True:
    user_input = input("User: ")
    
    style = random.choice(["formal", "casual", "funny"])
    location = "Karachi"
    context = WeatherContext(location=location, style=style)

    result = Runner.run_sync(
        agent,
        user_input,
        run_config=config,
        context=context
    )

    print(f"AI ({style} mode): {result.final_output}")
