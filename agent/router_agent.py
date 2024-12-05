# router_agent.py
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
query_memory = None
prompt = None

def initialize_router_agent(llm_instance, chat_memory_instance):
    global llm, chat_memory, prompt
    llm = llm_instance
    chat_memory = chat_memory_instance

    system_prompt = """You are an intelligent query classification system for an e-commerce platform.
    Your role is to accurately categorize incoming customer queries into one of two categories:

    1. product_review: 
       - Queries about product options, features, specifications, or capabilities
       - Questions about product prices and availability
       - Requests for product review, customer review, top review or comparisons
       - Questions about product warranties or guarantees
       - Inquiries about product shipping or delivery
       - Questions about product compatibility or dimensions
       - Requests for recommendations between products

    2. generic:
       - General customer service inquiries
       - Account-related questions
       - Technical support issues
       - Website navigation help
       - Payment or billing queries
       - Return policy questions
       - Company information requests

    INSTRUCTIONS:
    - Analyze the input query carefully
    - Respond ONLY with either "product_review" or "generic"
    - Do not include any other text in your response
    - If unsure, classify as "generic"

    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    logger.info("Router agent initialized successfully")


def classify_query(query):
    try:
        # Create chain with memory
        chain = prompt | llm

        # Classify the query
        response = chain.invoke({"input": query})
        category = response.content.strip().lower()
        
        # Validate category
        if category not in ["product_review", "generic"]:
            category = "generic"  # Default fallback


        # Add classification result to chat history
        if chat_memory:
            chat_memory.save_context(
                {"User": query},  # User's input
                {"AI": category}  # AI's response
            )

        
        # if chat_memory and hasattr(chat_memory, 'chat_memory'):
        #     chat_memory.save_context(
        #         {"AI": f"Query classified as: {category}"
        #         } 
        #     )
        #     # chat_memory.add_ai_message(AIMessage(content=f"Query classified as: {category}"))
            # logger.info(f"Ending In classify_query chat history: {chat_memory}")
        
        logger.info(f"Query: {query}")
        logger.info(f"Classification: {category}")
        print("**** in router agent****")
        print("query :", query)
        print("category :", category)

        return category

    except Exception as e:
        print(f"Error in routing: {str(e)}")
        return "generic"  # Default fallback on error


def get_classification_history():
    """Retrieve classification history from memory"""
    if chat_memory and hasattr(chat_memory, 'chat_memory'):
        return chat_memory.chat_memory.messages
    return []


def clear_context():
    """Clear all memory contexts"""
    if chat_memory:
        chat_memory.clear()
    logger.info("Router agent context cleared")