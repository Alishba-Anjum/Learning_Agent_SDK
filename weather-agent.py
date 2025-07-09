import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")



external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = 'gemini-2.0-flash',
    openai_client=external_client
)

config = RunConfig(
    model = model,
    model_provider= external_client,
    tracing_disabled = True
)

weather_agent = Agent(
    name = "Weather Agent",
    instructions = "You are a weather agent. Provide current weather information for any location. if someone asked any other question, politely decline and ask them to ask about the weather.",
    model = model
)


while True:
    user_input = input('User: ')
    result = Runner.run_sync(weather_agent  , user_input, run_config=config)
    print(f"Weather Agent: {result.final_output}")
