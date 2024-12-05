# planning_agent.py
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import SimpleMemory
from langchain.schema import HumanMessage, AIMessage
import agent.router_agent as router_agent
import agent.product_review_agent as product_review_agent
import agent.generic_agent as generic_agent
import agent.composer_agent as composer_agent
import logging

# Set httpx (HTTP request) logging to WARNING or ERROR level

logging.getLogger("httpx").setLevel(logging.WARNING) 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None
agent = None

def initialize_planning_agent(llm_instance, chat_memory_instance, query_memory_instance):
    global llm, chat_memory, query_memory, agent
    
    llm = llm_instance
    chat_memory = chat_memory_instance
    query_memory = query_memory_instance
    
    # Initialize agents
    router_agent.initialize_router_agent(llm, chat_memory)
    product_review_agent.initialize_product_review_agent(llm, chat_memory)
    generic_agent.initialize_generic_agent(llm, chat_memory)

    
    tools = [
        Tool(
            name="route_query",
            func=route_query,
            description="Determine query type. Returns either 'product_review' or 'generic'"
        ),
        Tool(
            name="get_product_info",
            func=get_product_info,
            description="Use this to get product related data such as options, features, prices, availability, or reviews"
        ),
        Tool(
            name="handle_generic_query",
            func=handle_generic_query,
            description="Use this to get response to user queries which are generic"
            # description="Use this to get response to user queries which are generic and where the retrieval of product details are not required"
        ),
        Tool(
            name="compose_response",
            func=compose_response,
            description="Use this to only format the response. After this step, return the formatted response to main.py"
        )
    ]
    
    
    system_prompt = """
You are an efficient and helpful AI planning agent. Do not assume anything. Always use route_query to determine the router_response. The router_response is the variable which has the user query type. Always refer to this variable router_response.
If router_response is 'generic', use handle_generic_query to process the user query and then use compose_response to format the response. 
If router_response is 'product_review', use get_product_info to retrieve product-related data and then use compose_response to format the response.

Example:

    User: Okay , i want to buy a phone. What buying options do you have ?
    Thought: I will use route_query to understand if query is product_review or generic
    Action: route_query
    Observation: product_review
    Thought: This query is actually a product review request, not a generic query. I will use get_product_info to get appropriate response.
    Action: get_product_info
    Action Input: User query: Okay , i want to buy a phone . What options do you have ?
    Observation: ok
    Thought:I have got the final answer. I will use compose_responses to format the response.
    Action: compose_responses
    Final Answer: ok
    """

    logger.info("Planning agent : Trying tp initialize agent")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=chat_memory,
        system_message=system_prompt,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        max_iterations=2
    )
    logger.info("Planning agent initialized successfully")

def load_session_memory(session_id, user_sessions):
    global chat_memory, query_memory
    try:
        # Load chat memory
        chat_memory = user_sessions[session_id]["chat_memory"]
        logger.info("Planning agent: Chat memory loaded successfully")
        
        # Load query memory
        query_memory = user_sessions[session_id]["query_memory"]
        logger.info("Planning agent: Query memory loaded successfully")

    except KeyError as e:
        logger.error(f"Error loading session memory: {str(e)}. Session ID: {session_id} may not exist.")
        raise  # Optionally re-raise the error if you want it to propagate

    except Exception as e:
        logger.error(f"Unexpected error while loading session memory: {str(e)}")
        raise

    
def route_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    router_response = router_agent.classify_query(original_query)
    return router_response


def get_product_info(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    response = product_review_agent.process(original_query)

    return {
        "intermediate_steps": [],
        "output": response,
        "action": "Final Answer",
        "action_input": response
    }

def handle_generic_query(query):
    # Get original query from memory if needed
    original_query = query_memory.memories.get('original_query', query)
    response = generic_agent.process(original_query)
    return {
        "intermediate_steps": [],
        "output": response,
        "action": "Final Answer",
        "action_input": response
    }


def compose_response(response):
    return composer_agent.compose_response(response)

def execute(query, session_id, user_sessions):
    load_session_memory(session_id, user_sessions)
    
    try:
        # Store original query
        query_memory.memories['original_query'] = query
        # Add the user query to chat memory
        # chat_memory.add_message(HumanMessage(content=query))
        result = agent.run(
            f"Process this user query: {query}"
        )

        chat_memory.save_context(
            {"User": query},  # User's input
            {"AI": result}  # AI's response
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in planning agent: {str(e)}")
        return f"Error in planning agent: {str(e)}"


def clear_context():
    if chat_memory:
        chat_memory.clear()
        logger.info("Planning agent: Chat memory cleared successfully")
    if query_memory:
        query_memory.memories.clear()
        logger.info("Planning agent: Query memory cleared successfully")
    product_review_agent.clear_context()
    generic_agent.clear_context()
