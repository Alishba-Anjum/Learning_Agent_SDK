import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, handoff, function_tool
from agents.run import RunConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")    
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model='gemini-2.0-flash',
    openai_client=external_client
)
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_joke() -> str:
    print("get_joke tool called")
    return "I'm a bot, I don't have a sense of humor."


Joke_Agent = Agent(
    name='Joke Agent',
    instructions='You are a Joke Agent. You can provide jokes to lighten the mood. Use the get_joke tool to fetch a joke.',
    model=model,
    tools=[get_joke]
)

while True:
    user_input=input('User: ')
    result = Runner.run_sync(
        Joke_Agent,
        user_input,
        run_config=config
    )

    print(f"Joke Agent: {result.final_output}")