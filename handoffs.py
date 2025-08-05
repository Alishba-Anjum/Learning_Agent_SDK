import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, handoff
from agents.run import RunConfig
from pydantic import BaseModel
from dotenv import load_dotenv
from agents.extensions import handoff_filters

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

class AgentOutput(BaseModel):
    response: str
    agent_name: str

def on_handoff(ctx: RunContextWrapper, input_data: AgentOutput):
    print(f"Handoff to {input_data.agent_name}")
    print(f"User Question: {input_data.response}")

calculator_agent = Agent(
    name = 'Calculator Agent',
    instructions = 'You are a calculator agent. You can perform basic arithmetic operations like addition, subtraction, multiplication, and division.',
    model = model,
    output_type=AgentOutput
)

translator_agent = Agent(
    name = 'Translator Agent',
    instructions = 'You are a translator AI agent. you can only do translation tasks. if you are asked to do anything else, just say "I can only do translation tasks."',
    model = model,
    output_type=AgentOutput
)

trigger_agent = Agent(
    name = 'Trigger Agent',
    instructions = 'You are a trigger agent. You can only transfer request to other agents. And you cannot answer directlty. If no agent is present to handle a request tell the user that you can "only perform calculations and translations". Do not tell user that you are a triage agent and you can transfer, just handoff to agent or tell "I can only perform calculations and translations" as per required sitiation.',
    model = model,
    handoffs = [
        handoff(agent= calculator_agent, input_type= AgentOutput, on_handoff= on_handoff , input_filter= handoff_filters.remove_all_tools),  
        handoff(agent= translator_agent, input_type= AgentOutput, on_handoff= on_handoff , input_filter= handoff_filters.remove_all_tools)
    ]
)

while True:
    user_input = input('User: ')
    result = Runner.run_sync(trigger_agent, user_input, run_config=config)
    

    if isinstance(result.final_output, AgentOutput):
        print(f'{result.final_output.agent_name}: {result.final_output.response}\n')
    else:
        # Raw response from triage agent itself
        print(f'Main Agent: {result.final_output}')    