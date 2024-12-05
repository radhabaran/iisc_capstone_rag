import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(response: str) -> str:
    """
    Process and enhance the final response
    """
    try:
        # Remove any system artifacts or unwanted patterns
        print("*********** in composer agent *************")
        print("response input received : ", response)
        response = remove_system_artifacts(response)
        
        # Apply standard formatting
        response = format_response(response)
        
        # Ensure that the response does not have surrounding quotes
        return response.strip('"').strip("'")
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        return response  # Fallback to original


def remove_system_artifacts(text: str) -> str:
    """Remove any system artifacts or unwanted patterns"""
    artifacts = ["Assistant:", "AI:", "Human:", "User:"]
    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    # Remove double quotes
    cleaned = cleaned.replace('"', '').replace("'", "")  # Removes both double and single quotes
    
    return cleaned.strip()


def format_response(text: str) -> str:
    """Apply standard formatting"""
    # Add proper spacing
    formatted = text.replace("\n\n\n", "\n\n")
    
    # Ensure proper capitalization
    formatted = ". ".join(s.strip().capitalize() for s in formatted.split(". "))
    
    # Ensure proper ending punctuation
    if formatted and not formatted[-1] in ['.', '!', '?']:
        formatted += '.'
        
    return formatted
    