import os
from agents import Runner, Agent , OpenAIChatCompletionsModel, AsyncOpenAI , RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import chainlit as cl

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

@cl.on_chat_start
async def on_chat_start():

    cl.user_session.set('history', [])
    
    await cl.Message(
        content="Welcome to the Weather Agent! Ask me about the weather in any location.",
        author="Weather Agent"
    ).send() 
    

@cl.on_message
async def handle_message(message: cl.Message):
    if not message.content:
        await cl.Message(content="Please provide a location to get the weather information.").send()
        return
    history = cl.user_session.get('history', [])
    history.append({"role": "user", "content": message.content})
     
   
    msg = cl.Message(content=f"Weather Agent:  {message.content}", author="Weather Agent")
    await msg.send()

    result =  Runner.run_streamed(
        weather_agent,
        history,
         run_config=config
        )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)


    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set('history', history)
    
    # await cl.Message(content=f"Weather Agent:  {result.final_output}").send()