import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, SimpleMemory
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os
import agent.planning_agent as planning_agent
import logging
import uuid
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
user_sessions = {}


def create_session():
    """Create a new session with unique identifiers for chat and query memory."""
    session_id = str(uuid.uuid4())
    logger.info("App.py : session_id created successfully :")
    print("session_id :", session_id)
    return session_id
    

def initialize_llm():
    global llm

    api_key = os.environ['OA_API']           
    os.environ['OPENAI_API_KEY'] = api_key
    
    load_dotenv()
        
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    logger.info("LLM initialized successfully")


def initialize_user_session(session_id):
    """Initialize components for a specific user session"""

    if session_id not in user_sessions:
        chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
        query_memory = SimpleMemory()
  
        # Initialize the planning agent with both memories for the new session
        planning_agent.initialize_planning_agent(llm, chat_memory, query_memory)

        user_sessions[session_id] = {
            "chat_memory": chat_memory,  # Store chat memory
            "query_memory": query_memory
        }
        logger.info(f"Initialized components for session: {session_id}")
        
    return user_sessions[session_id]


def count_tokens(text):
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(str(text)))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

        
def create_summary(history, llm):
    combined_history = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history])
    
    summary_prompt = f"""Summarize our conversation so far in about 100 words. 
    Start with: "Let me summarize our conversation so far:"
    Keep the most important points and context for continuing our discussion.
    
    Conversation to summarize:
    {combined_history}"""
    
    try:
        summary = llm.predict(summary_prompt)
        return summary
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}")
        return None
        

def process_query(query, history, session_id):
    try:
        logger.info(f"Processing query for session: {session_id}")

        # Initialize session if it doesn't exist
        if session_id not in user_sessions:
            initialize_user_session(session_id)
                
        chat_memory = user_sessions[session_id]["chat_memory"]
        query_memory = user_sessions[session_id]["query_memory"]

        # Check token count
        total_tokens = count_tokens(str(history))
        logger.info(f"Token count ({total_tokens})")

        # If tokens exceed threshold (e.g., 5000), create summary
        if total_tokens > 5000:
            logger.info(f"Token count ({total_tokens}) exceeded threshold, creating summary...")
            summary = create_summary(history, llm)
            if summary:
                logger.info(f"Summary created: {len(summary)} chars, {count_tokens(summary)} tokens")
                # Clear existing memory
                chat_memory.clear()
                # Store summary as AI message
                chat_memory.save_context(
                    {"User": "Could you summarize our conversation so far?"},
                    {"AI": summary}
                )
                # Keep the last 2-3 exchanges for immediate context
                recent_messages = history[-3:]
                for human_msg, ai_msg in recent_messages:
                    chat_memory.save_context(
                        {"User": human_msg},
                        {"AI": ai_msg}
                    )
                logger.info("Chat history summarized successfully")

        # # Restore chat history from Gradio's history
        # if history:
        #     for human_msg, ai_msg in history:
        #         chat_memory.save_context(
        #             {"User": human_msg},  # User's input
        #             {"AI": ai_msg}  # AI's response
        #         )

        # Store original query in query memory
        query_memory.memories['original_query'] = query
        
        # Execute query through planning agent, passing session_id and user_sessions
        response = planning_agent.execute(query, session_id, user_sessions)
        
        # Add current interaction to chat memory
        chat_memory.save_context(
            {"User": query},  # User's input
            {"AI": response}  # AI's response
        )

        return response

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(f"Error details: {str(e)}")

        return error_msg


def clear_context(session_id=None):
    """Clear the context for a specific user session."""
    if session_id in user_sessions:
        planning_agent.clear_context()  # Clear the planning agent context
        user_sessions[session_id]["chat_memory"].clear()
        user_sessions[session_id]["query_memory"].memories.clear()
        logger.info(f"Cleared context for session: {session_id}")
    return [], ""


def create_gradio_app(session_id):
    from interface import create_interface
    return create_interface(process_query, clear_context, session_id)


def main():
    """Main application entry point"""
    try:
        session_id = create_session()
        print("in main session_id :", session_id)
        initialize_llm()
        initialize_user_session(session_id)
        
        logger.info("Starting Gradio app")
        app = create_gradio_app(session_id)
        app.queue()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()