from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()       # Load environment variables from a .env file

# Initialize the model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)