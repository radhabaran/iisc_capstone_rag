# product_review_agent.py
# ***********************************************************************************************
# Instruction for using the program
# ***********************************************************************************************
# Please make sure the embeddings.npy file is available in data folder
# Please make sure the documents.pkl file is available in data folder
# Please set the path appropriately inside the program. You will find the below two statements 
# where you need to mention the correct path name.
# embedding_path = '/workspaces/IISC_cap_langchain/data/embeddings.npy'
# documents_path = '/workspaces/IISC_cap_langchain/documents.pkl'
# ***********************************************************************************************

import openai
import numpy as np
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
# from langchain.memory import ConversationBufferMemory, SimpleMemory
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import warnings
import os

warnings.filterwarnings("ignore")
import pickle
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
chat_memory = None


def initialize_product_review_agent(llm_instance, memory_instance):
    """Initialize the product review agent with LLM and memory instances"""
    global llm, chat_memory

    llm = llm_instance
    chat_memory = memory_instance


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects containing text content to split.

    Returns:
        list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk in characters
        chunk_overlap=300,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks  # Return the list of split text chunks

    
def initialize_vectorstore(documents, embeddings_list, vectorstore_path: str = '/home/user/app/docs/chroma/') -> Chroma:
    
    vectordb = None
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    # To list all files and directories in current directory:
    print("\nContents of current directory:")
    print(os.listdir(current_dir))
    try:
        # Create directory if it doesn't exist
        logger.info("Creating directory if it doesn't exist...")
        os.makedirs(vectorstore_path, exist_ok=True)
        logger.info(f"Directory checked/created at: {vectorstore_path}")
        
        # Check if vectorstore exists
        if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
            logger.info(f"Loading existing vectorstore from {vectorstore_path}")
            
            vectordb = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embeddings_list
            )
            collections = vectordb._client.list_collections()
            logger.info(f"Existing collections: {collections}")
            
        else:

            # Remove the old database files if any
            shutil.rmtree('./docs/chroma', ignore_errors=True)

            # Create new vectorstore
            logger.info(f"Creating new vectorstore at {vectorstore_path}")
            
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embeddings_list,
                persist_directory=vectorstore_path,
            )
            # Persist the vectorstore
            vectordb.persist()
            # Create a zip file of the vectorstore directory
            shutil.make_archive('vectorstore', 'zip', vectorstore_path)
            logger.info("Vectorstore created and persisted successfully")
            collections = vectordb._client.list_collections()
            logger.info(f"New collections: {collections}")
        
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {str(e)}")
        raise Exception(f"Failed to initialize vectorstore: {str(e)}")
    
    return vectordb

    

def process(query):

    System_Prompt = """
    Role and Capabilities:
You are an AI customer service specialist for Amazon. You respond strictly based on the context provided and /
from the previous chat history. Wheneve user mentions Amazon, you refer to strictly to local knowledge base. / 
Your primary functions are: 
1. Providing accurate product information including cost, availability, features, top review or user rating. Treat top review, user rating, user feedback are all same request.
2. Handling delivery-related queries
3. Addressing product availability
4. Offering technical support for electronics

Core Instructions:
1. Product Information:
   - Provide detailed specifications and features based only on the provided context.
   - Compare similar products when relevant only if they appear in the provided context.
   - Only discuss products found in the provided context.
   - Highlight key benefits and limitations found in the context.
   - Include top reviews or user ratings only if available in the context.

2. Price & Availability:
   - Quote exact prices and stock availability directly from the provided context.
   - Explain any pricing variations or discounts only if stated in the context.
   - Provide clear stock availability information only if stated in the context.
   - Mention delivery timeframes only when available in the context.

3. Query Handling:
   - Address the main query first, then provide additional relevant information from the context.
   - For multi-part questions, structure answers in bullet points
   - If information is missing from context, explicitly state this
   - Suggest alternatives when a product is unavailable

Communication Guidelines:
1. Response Structure:
   - Start with a direct answer to the query based solely on the provided context.
   - Provide supporting details and context from the provided information only.
   - End with a clear next step or call to action
   - Include standard closing: "Thank you for choosing Amazon. Is there anything else I can help you with?"

2. Tone and Style:
   - Professional yet friendly
   - Clear and jargon-free language
   - Empathetic and patient
   - Concise but comprehensive

Limitations and Restrictions:
1. Provide information present only in the given context.
2. Do not provide answers from memory; rely exclusively on the provided context.
2. Clearly state when information is not available in the context.
3. Never share personal or sensitive information
4. Don't make promises about delivery times unless explicitly stated in context

Error Handling:
1. Out of Scope: "While I can't assist with [topic], I'd be happy to help you other products if you like."
2. Technical Issues: "I apologize for any inconvenience. Could you please rephrase your question or provide more details?"

Response Format:
1. For product queries:
   - Product name and model
   - Price and availability
   - Key features
   - Top review or user rating
   - Comparison among similar products (example : cell phone with cell phone, not with cell phone accesories)
   - Recommendations if relevant

2. For service queries:
   - Current status
   - Next steps
   - Timeline (if available)
   - Contact options

Remember: Always verify information against the provided context or in the previous chat history before /
responding. Don't make assumptions or provide speculative information.

    """

    # Get existing chat history from memory
    chat_history = ""
    if chat_memory:
        messages = chat_memory.chat_memory.messages
        chat_history = "\n".join(f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in messages)

    # Check if embeddings already exist
    embedding_path = './data/embeddings.npy'
    documents_path = './documents.pkl'
    file_path = './data/cleaned_dataset_full.csv'

    try:
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found at: {documents_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Initialize the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Modify the get_embedding function to use LangChain's OpenAIEmbeddings
    # def get_embedding(text, engine="text-embedding-ada-002"):
    #     return embeddings.embed_query(text)

        
    # Load the DataFrame
    dataframed = pd.read_csv(file_path)
    # dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(f"{col}: {val}" for col, val in row.items()), axis=1)

    # Combine all rows into a single string
    combined_text = ' '.join(dataframed['combined'].tolist())

    # Create a single Document object
    single_document = Document(page_content=combined_text)

    # Call the split_text function with the single document
    split_chunks = split_text([single_document])

    # Initialize vectorstore - call custom function initialize_vectorstore
    vectordb = initialize_vectorstore(
        documents=split_chunks,
        embeddings_list=embeddings,  
        vectorstore_path='/home/user/app/docs/chroma/'
    )

    # *****************************************************************************************
    # *******************************  Retrieval Method# 1 *************************************
    # *****************************************************************************************
    # similarity_search
    
    # results = vectordb.similarity_search_with_relevance_scores(query, k=3)
    
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     logger.warning("No relevant documents retrieved. Returning fallback response.")
    #     context = "No relevant information was found in the database."
    # else:
    #     logger.info(f"Retrieved {len(results)} documents.")
    #     context= "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # ****************************** Retrieval Method# 1 Ends ***********************************

    # *****************************************************************************************
    # *******************************  Retrieval Method# 2 *************************************
    # *****************************************************************************************
    # # Create a retriever (Maximum marginal relevance retrieval)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k":5})

    # Use the retriever to fetch relevant documents
    results = retriever.invoke(query)

    # Print results from reriever
    print("\n***************** Reriever Begins********************")
    print("results: ", results)
    print("\n***************** Retriever Ends ********************")

    if len(results) == 0:
        print("Unable to find matching results.")
        logger.warning("No relevant documents retrieved. Returning fallback response.")
        context = "No relevant information was found in the database."
    else:
        logger.info(f"Retrieved {len(results)} documents.")
        # context = "\n\n---\n\n".join([doc.page_content for doc in results])
        context = "\n\n".join([doc.page_content for doc in results])
        
        
    # ****************************** Retrieval Method# 2 Ends ***********************************

    # *****************************************************************************************
    # *******************************  Retrieval Method# 3 *************************************
    # *****************************************************************************************
    # Similarity score threshold retrieval
    
    # retriever = vectordb.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    # )

    # # Use the retriever to fetch relevant documents
    # results = retriever.invoke(query)

    # # Print results from reriever
    # print("\n***************** Reriever Begins********************")
    # print("results: ", results)
    # print("\n***************** Retriever Ends ********************")

    # if len(results) == 0:
    #     print("Unable to find matching results.")
    #     logger.warning("No relevant documents retrieved. Returning fallback response.")
    #     context = "No relevant information was found in the database."
    # else:
    #     logger.info(f"Retrieved {len(results)} documents.")
    #     context = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # ****************************** Retrieval Method# 3 Ends ***********************************

    # *****************************************************************************************
    # *******************************  Retrieval Method# 4 *************************************
    # *****************************************************************************************
    # Specifying top k
    
    # retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # # Use the retriever to fetch relevant documents
    # results = retriever.invoke(query)

    # # Print results from reriever
    # print("\n***************** Reriever Begins********************")
    # print("results: ", results)
    # print("\n***************** Retriever Ends ********************")

    # if len(results) == 0:
    #     print("Unable to find matching results.")
    #     logger.warning("No relevant documents retrieved. Returning fallback response.")
    #     context = "No relevant information was found in the database."
    # else:
    #     logger.info(f"Retrieved {len(results)} documents.")
    #     context = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # ****************************** Retrieval Method# 4 Ends ***********************************
    
    # Include chat history in the prompt for context
    structured_prompt = f"""
    <context>
    {context}
    </context>

    Chat history:
    {chat_history}
    
    Current Query:
    {query}
    """

    print("structured prompt created :", structured_prompt)
    print('*' * 100)
    # Create messages for the chat model
    messages = [
        {"role": "system", "content": System_Prompt},
        {"role": "user", "content": structured_prompt}
    ]

    # For chat completion, you can use LangChain's ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = chat_model.invoke(messages).content

    # Update memory with structured messages
    if chat_memory:
        chat_memory.save_context(
            {"User": query},  # User's input
            {"AI": response}  # AI's response
        )


    logger.info(f"Successfully processed query: {query}")
    print("response returned by product_review_agent", response)
    return response


def clear_context():
    """Clear the conversation memory"""
    if chat_memory:
        chat_memory.clear()
        logger.info("Conversation context cleared")