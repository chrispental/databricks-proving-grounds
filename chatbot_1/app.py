import gradio as gr
import logging
import os
from textwrap import dedent
from typing import Iterator
from agno.agent import Agent, RunResponseEvent
from agno.models.openrouter import OpenRouter
from agno.tools.thinking import ThinkingTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.tavily import TavilyTools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the thinking agent
thinking_agent = Agent(
    model=OpenRouter(id="google/gemini-2.0-flash-001"),
    tools=[
        ThinkingTools(add_instructions=True),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        ),
        TavilyTools(),
    ],
    instructions=dedent("""
        You are a helpful and knowledgeable AI assistant.

        Engage in open-ended conversation, answer questions, provide explanations, and assist with a wide variety of topics.
        Be clear, concise, and friendly. If you need more information, ask follow-up questions.
        Support users in exploring ideas, solving problems, or just having a conversation.
        """),
    add_datetime_to_instructions=True,
    stream_intermediate_steps=True,
    show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,  # Enable conversation memory
    num_history_responses=10,       # Number of previous exchanges to remember
)

def query_llm(message, history):
    """
    Query the agno agent with the given message and chat history using streaming.
    `message`: str - the latest user input.
    `history`: list of tuples - Gradio-style chat history [(user_msg, assistant_msg), ...].
    """
    if not message.strip():
        yield "ERROR: The question should not be empty"
        return

    try:
        logger.info(f"Processing message with agno agent: {message}")
        
        # Get streaming response from the thinking agent
        response_stream: Iterator[RunResponseEvent] = thinking_agent.run(
            message,
            stream=True
        )
        
        # Stream the response
        accumulated_response = ""
        for event in response_stream:
            if hasattr(event, 'content') and event.content:
                accumulated_response += event.content
                yield accumulated_response
            elif hasattr(event, 'delta') and event.delta:
                accumulated_response += event.delta
                yield accumulated_response
        
        # Ensure we have a final response
        if not accumulated_response:
            yield "No response received from agent"
            
    except Exception as e:
        logger.error(f"Error querying agno agent: {str(e)}", exc_info=True)
        yield f"Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=query_llm,
    title="Agno AI Assistant",
    description=(
        "An intelligent AI assistant powered by Agno framework with thinking capabilities, "
        "financial data access (YFinance), and web search (Tavily). "
        "Ask questions, explore ideas, get stock information, or have a conversation!"
    ),
    examples=[
        "What's the current stock price of Apple?",
        "Explain quantum computing in simple terms",
        "What are the latest news about Tesla?",
        "Help me understand machine learning concepts",
        "Search for recent AI developments"
        "Tell me a story about Poland's history",
        "What are the analyst recommendations for Microsoft?",
        "What is the weather like in New York today?",
    ],
)

if __name__ == "__main__":
    demo.launch()