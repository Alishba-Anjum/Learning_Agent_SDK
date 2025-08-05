import os
from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, function_tool
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
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

@function_tool
def add(a: float, b: float) -> float:
    '''
    Add given two numbers.

    args:
        a: First number to add
        b" Second number to add
    '''

    print("add tool called")
    return a + b 

@function_tool
def substract(a:float, b:float) -> float:
    '''
    SUbstract given two numbers.

    args:
        a: First number to substract
        b: Second number to substract
    '''
    print("substract tool called")
    return a - b

@function_tool
def multiply(a:float, b:float) -> float:
    '''
    Multiply given two numbers.

    args:
        a: First number to multiply
        b: Second number to multiply
    '''
    print("multiply tool called")
    return a * b

@function_tool
def divide(a:float, b:float) -> float:
    '''
    Divide given two numbers.

    args:
        a: dividend
        b: divisor
    '''
    print("divide tool called")
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


calculator_agent = Agent(
    name = 'Calculator Agent',
    instructions = 'You are a calculator agent. You can perform basic arithmetic operations like addition, subtraction, multiplication, and division. Use the tools provided to perform calculations.',
    model = model,
    tools = [add, substract, multiply, divide]
)

while True:
    result = Runner.run_sync(calculator_agent, input('User: '), run_config=config)
    if result.final_output:
        print(f"Calculator Agent: {result.final_output}\n") 

        