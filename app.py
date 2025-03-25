import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import BingGroundingTool

###############################################################################
#                               ENVIRONMENT SETUP
###############################################################################
load_dotenv()

# Load environment variables
API_KEY = os.getenv("api_key")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Validate required environment variables
if not all([API_KEY, PROJECT_CONNECTION_STRING, BING_CONNECTION_NAME, MODEL_DEPLOYMENT_NAME, MODEL_API_VERSION, AZURE_ENDPOINT]):
    raise ValueError("Missing one or more required environment variables.")

###############################################################################
#                               CLIENT INITIALIZATION
###############################################################################
# Azure OpenAI Client
az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=MODEL_DEPLOYMENT_NAME,
    model=MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY
)

# AI Project Client
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=PROJECT_CONNECTION_STRING,
)

# Retrieve Bing connection
bing_connection = project_client.connections.get(connection_name=BING_CONNECTION_NAME)
conn_id = bing_connection.id

###############################################################################
#                               HELPER FUNCTIONS
###############################################################################
async def execute_tool(stock_name: str, agent_name: str, instructions: str) -> str:
    """
    Generic function to execute a tool using BingGroundingTool and Azure AI.
    """
    print(f"[{agent_name}] Executing tool for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name=agent_name,
        instructions=instructions,
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    try:
        # Create a thread and send the user query
        thread = project_client.agents.create_thread()
        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=f"{instructions} for {stock_name}."
        )

        # Process the run and fetch messages
        project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
        messages = project_client.agents.list_messages(thread_id=thread.id)
        return messages["data"][0]["content"][0]["text"]["value"]

    except KeyError as e:
        print(f"[Error] Missing key in response: {e}")
        return "Error: Unable to fetch data due to missing information."
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
        return "Error: Unable to fetch data due to an unexpected issue."
    finally:
        # Clean up the agent
        project_client.agents.delete_agent(agent.id)

###############################################################################
#                               TOOL FUNCTIONS
###############################################################################
async def stock_price_trends_tool(stock_name: str) -> str:
    instructions = "Retrieve real-time stock prices, changes over the last few months, and summarize market trends"
    return await execute_tool(stock_name, "stock_price_trends_tool_agent", instructions)

async def news_analysis_tool(stock_name: str) -> str:
    instructions = "Retrieve the latest news articles and summaries"
    return await execute_tool(stock_name, "news_analysis_tool_agent", instructions)

async def market_sentiment_tool(stock_name: str) -> str:
    instructions = "Analyze general market sentiment and user opinions"
    return await execute_tool(stock_name, "market_sentiment_tool_agent", instructions)

async def analyst_reports_tool(stock_name: str) -> str:
    instructions = "Find recent analyst reports, price targets, or professional opinions"
    return await execute_tool(stock_name, "analyst_reports_tool_agent", instructions)

async def expert_opinions_tool(stock_name: str) -> str:
    instructions = "Collect expert opinions or quotes from industry leaders"
    return await execute_tool(stock_name, "expert_opinions_tool_agent", instructions)

###############################################################################
#                               AGENT FUNCTIONS
###############################################################################
async def stock_price_trends_agent(stock_name: str) -> str:
    return await stock_price_trends_tool(stock_name)

async def news_analysis_agent(stock_name: str) -> str:
    return await news_analysis_tool(stock_name)

async def market_sentiment_agent(stock_name: str) -> str:
    return await market_sentiment_tool(stock_name)

async def analyst_reports_agent(stock_name: str) -> str:
    return await analyst_reports_tool(stock_name)

async def expert_opinions_agent(stock_name: str) -> str:
    return await expert_opinions_tool(stock_name)

###############################################################################
#                               ASSISTANT AGENTS
###############################################################################
stock_trends_agent_assistant = AssistantAgent(
    name="stock_trends_agent",
    model_client=az_model_client,
    tools=[stock_price_trends_agent],
    system_message=(
        "You are the Stock Price Trends Agent. "
        "You fetch and summarize stock prices, changes over the last few months, and general market trends. "
        "Do NOT provide any final investment decision."
    )
)

news_agent_assistant = AssistantAgent(
    name="news_agent",
    model_client=az_model_client,
    tools=[news_analysis_agent],
    system_message=(
        "You are the News Agent. "
        "You retrieve and summarize the latest news stories related to the given stock. "
        "Do NOT provide any final investment decision."
    )
)

sentiment_agent_assistant = AssistantAgent(
    name="sentiment_agent",
    model_client=az_model_client,
    tools=[
        market_sentiment_agent,
        analyst_reports_agent,
        expert_opinions_agent
    ],
    system_message=(
        "You are the Market Sentiment Agent. "
        "You gather overall market sentiment, relevant analyst reports, and expert opinions. "
        "Do NOT provide any final investment decision."
    )
)

decision_agent_assistant = AssistantAgent(
    name="decision_agent",
    model_client=az_model_client,
    system_message=(
        "You are the Decision Agent. After reviewing the stock data, news, sentiment, analyst reports, "
        "and expert opinions from the other agents, you provide the final investment decision. "
        "In the final decision, make a call to either Invest or Not. Also provide the current stock price. "
        "End your response with 'Decision Made' once you finalize the decision."
    )
)

###############################################################################
#                        TERMINATION & TEAM CONFIGURATION
###############################################################################
text_termination = TextMentionTermination("Decision Made")
max_message_termination = MaxMessageTermination(15)
termination = text_termination | max_message_termination

investment_team = RoundRobinGroupChat(
    [
        stock_trends_agent_assistant,
        news_agent_assistant,
        sentiment_agent_assistant,
        decision_agent_assistant,
    ],
    termination_condition=termination
)

###############################################################################
#                                   MAIN
###############################################################################
async def main():
    try:
        stock_name = "tata motors"
        await Console(
            investment_team.run_stream(
                task=f"Analyze stock trends, news, and sentiment for {stock_name}, plus analyst reports and expert opinions, and then decide whether to invest."
            )
        )
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())