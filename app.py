from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
import streamlit as st

load_dotenv()

# Accessing environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm : ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash-exp',
    api_key=GEMINI_API_KEY
    )

@tool
def python_query_response(query: str) -> str:
    """Responds to Python programming-related queries and also provide examples and code snippets if user ask to give code related to any topic in markdown format.

    This function takes a Python-related query as input and provides an appropriate
    response, including explanations or examples where applicable.

    Args:
        query (str): The Python programming-related query.

    Returns:
        str: A detailed response to the query, including code examples or guidance.

    Raises:
        ValueError: If the query is not Python-related or is empty.
    """
    if not query:
        raise ValueError("Query cannot be empty. Please provide a valid Python-related query.")

    # Define a dictionary with sample Python-related questions and responses.
    responses = {
        "how to define a function": "You can define a function using the 'def' keyword. Example:\n\n" \
        "def my_function():\n    print('Hello, world!')",

        "what is a list in python": "A list is a mutable, ordered collection of items. Example:\n\n" \
        "my_list = [1, 2, 3, 'apple']",

        "how to handle exceptions": "You can handle exceptions using a try-except block. Example:\n\n" \
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero.')",

        "what is python": "Python is a high-level, interpreted programming language known for its simplicity and versatility. It is widely used in web development, data analysis, AI, and more.",
    }

    # Search for a predefined response or return a default response.
    response = responses.get(query.lower(), "I'm sorry, I couldn't find an answer to your query. Could you rephrase or ask something specific about Python?")

    return response

tools = [python_query_response,]


agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True 
)

st.title('Python Programming Assistant')
st.write('Welcome to the Python Programming Assistant😊! Please enter your Python-related query below.')
user_input = st.text_input('Enter Your Prompt')
#st.button('Submit')
if st.button('Submit'):
    response = agent.invoke(user_input)
    st.write(response['output'])