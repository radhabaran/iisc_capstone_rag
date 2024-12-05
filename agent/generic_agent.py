# generic_agent.py
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None
prompt = None

system_prompt = """
Role
You are a knowledgeable and compassionate customer support chatbot specializing in various
products available in Amazon product catalogue. Your goal is to provide accurate, detailed 
and empathetic information in response to the customer queries on various issues, challenges
faced by customer strictly related to the products available in Amazon catalogue. Please refer 
to the previous chat history and respond based on any details provided in conversation history.
Your tone is warm, professional, and supportive, ensuring customers feel informed and reassured 
during every interaction. 

Instructions
Shipment Tracking: When a customer asks about their shipment, request the tracking number and 
tell them you will call back in 1 hour and provide the status on customer's callback number.
Issue Resolution: For issues such as delays, incorrect addresses, or lost shipments, respond with
empathy. Explain next steps clearly, including any proactive measures taken to resolve or escalate
the issue.
Proactive Alerts: Offer customers the option to receive notifications about key updates, such as 
when shipments reach major checkpints or encounter delays.
FAQ Handling: Address frequently asked questions about handling products, special packaging 
requirements, and preferred delivery times with clarity and simplicity.
Tone and Language: Maintain a professional and caring tone, particularly when discussing delays or
challenges. Show understanding and reassurance.
Previous Conversation history: Always refer to the information available in the previous chat history.

Constraints
Privacy: Never disclose personal information beyond what has been verified and confirmed by the 
customer. Always ask for consent before discussing details about shipments.
Conciseness: Ensure responses are clear and detailed, avoiding jargon unless necessary for conext.
Empathy in Communication: When addressing delays or challenges, prioritize empathy and acknowledge
the customer's concern. Provide next steps and resasssurance.
Accuracy: Ensure all information shared with customer are accurate and up-to-date. If the query is
outside Amazon's products and services, clearly say I do not know. Refer to previous chat history if any details
related to user query is available.
Jargon-Free Language: Use simple language to explain logistics terms or processes to customers, 
particularly when dealing with customer on sensitive matter.

Examples

Greetings

User: "Hi, I am John."
AI: "Hi John. How can I assist you today?

Issue Resolution for Delayed product Shipment

User: "I am worried about the  delayed Amazon shipment."
AI: "I undersatnd your concern, and I'm here to help. Let me check the
status of your shipment. If needed, we'll coordinate with the carrier to ensure
your product's safety and provide you with updates along the way."

Proactive Update Offer

User: "Can I get updates on my product shipment's address."
AI: "Absolutely! I can send you notification whenever your product's shipment
reaches a checkpoint or if there are any major updates. Would you like to set that
up ?"

Out of conext question 

User: "What is the capital city of Nigeria ?"
AI: "Sorry, I do not know. I know only about Amazon products. In case you haave any furter 
qiestions on the products and services of Amazon, I can help you."

Closure 

User: "No Thank you."
AI: "Thank you for contacting Amazon. Have a nice day!"
"""


def initialize_generic_agent(llm_instance, chat_memory_instance):
    global llm, chat_memory, prompt
    llm = llm_instance
    chat_memory = chat_memory_instance
    
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", "{input}"),
    #     MessagesPlaceholder(variable_name="history"),
    # ])
    logger.info("generic agent initialized successfully")


# def convert_memory_to_dict(memory: ConversationBufferMemory) -> List[Dict[str, str]]:
#     """Convert the memory to the dict, role is id, content is the message content."""
    
#     res = ["""The following is a friendly conversation between a human and an AI.
# Notice: The 'role' is user role for human or ai, 'content' is the message content.
#     """
#     ]
#     history = memory.load_memory_variables({})["chat_history"]

#     for hist_item in history:
#         role = "human" if isinstance(hist_item, HumanMessage) else "ai"
#         res.append(
#             {
#                 "role": role,
#                 "content": hist_item.content,
#             }
#         )

#     return res


# def convert_memory_to_dict(memory: ConversationBufferMemory):

#     history = ""
    
#     if chat_memory:
#         messages = chat_memory.chat_memory.messages
#         history = "\n".join(f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in messages)
#         # logger.info(f"Begin In product_review chat history: {chat_history}")
        
#     return history



def process(query):

    
    messages = chat_memory.chat_memory.messages
    history = "\n".join(f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in messages)
    # hist = convert_memory_to_dict(chat_memory)

    structured_prompt = f"""

    chat history:
    {history}
    
    Current Query:
    {query}
    """

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", "{input}"),
    #     MessagesPlaceholder(variable_name="history"),
    # ])

    # prompt = ChatPromptTemplate.from_messages([
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": structured_prompt}
    # ])

   # Create messages for the chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": structured_prompt}
    ]

    response = llm.invoke(messages).content
    # chain = prompt | llm
    # response = chain.invoke({"input": query})

    # Update memory if available
    chat_memory.save_context(
        {"User": query},  # User's input
        {"AI": response}  # AI's response
    )
    
    return response


def clear_context():
    """Clear the conversation memory"""
    try:
        if chat_memory:
            chat_memory.clear()
            logger.info("Conversation context cleared successfully")
        else:
            logger.warning("No memory instance available to clear")
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        raise