import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


external_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)

model = OpenAIChatCompletionsModel(
    model = 'gemini-2.0-flash',
    openai_client=external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)


calculator_agent = Agent(
    name = 'Calculator Agent',
    instructions = 'You are a calculator agent. You can perform basic arithmetic operations like addition, subtraction, multiplication, and division.',
    model = model
)

while True:
    user_input = input('User: ')
    result = Runner.run_sync(calculator_agent, user_input, run_config=config)
    print(f'Calculator Agent: {result.final_output}')