"""
OpenAI API health check - verifies API availability and billing status.
Safe diagnostic test before dataset generation.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 5
TEST_PROMPT = "sadece OK yaz"


def check_api_health():
    """
    Perform minimal API health check.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False, "ERROR: OPENAI_API_KEY not found in environment variables.\n  Please set it in your .env file."
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return False, f"ERROR: Failed to initialize OpenAI client: {e}"
    
    # Make minimal API request
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": TEST_PROMPT}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0
        )
        
        # If we get here, API is working
        return True, "API OK — billing and quota active"
        
    except AuthenticationError as e:
        return False, "ERROR: Invalid API key.\n  Please check your OPENAI_API_KEY in .env file."
    
    except RateLimitError as e:
        return False, "ERROR: Rate limit exceeded.\n  Please wait a moment and try again, or check your rate limits at https://platform.openai.com/account/limits"
    
    except APIError as e:
        # Check for quota errors
        error_code = getattr(e, 'code', None)
        error_message = str(e).lower()
        
        if error_code == 'insufficient_quota' or 'insufficient_quota' in error_message or 'quota' in error_message:
            return False, "ERROR: Insufficient quota.\n  Your account has no remaining credits. Please add credits at https://platform.openai.com/account/billing"
        
        return False, f"ERROR: OpenAI API error.\n  {e}"
    
    except APIConnectionError as e:
        return False, f"ERROR: Connection to OpenAI API failed.\n  Please check your internet connection: {e}"
    
    except Exception as e:
        return False, f"ERROR: Unexpected error.\n  {e}"


def main():
    """Main execution function."""
    success, message = check_api_health()
    print(message)
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

