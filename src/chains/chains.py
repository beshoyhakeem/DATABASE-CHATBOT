from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase

from src.clinets.clinets import llm


# Create a chain to process SQL queries
def check_plotting():
  template = """
    You are a pandas ai expert at a company. You are interacting with a user who is asking you questions about the company's database.
    check that the user's question needs to be potting. Take the conversation history into account.
    
    
    Write only the Boolean Response and nothing else. Do not wrap the Boolean Response in any other text, not even backticks or space.
    
    For example:
    Question: Plot a histogram of countries showing GDP, using different colors for each bar.
    Boolean Response:True
    Question: Create a line chart of sales over time?
    Boolean Response:True
    Question: How many employees are there?
    Boolean Response:False
    
    Your turn:
    
    Question: {question}
    Boolean Response:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  return (
    prompt
    | llm
    | StrOutputParser()
  )

def get_sql_table(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, extract the table name that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the table name and nothing else. Do not wrap the table name in any other text, not even backticks or space.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query:Track
    Question: Name 10 artists
    SQL Query:Artist
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
    
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )



def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
 
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  


  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  try:
        return chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
  
  except Exception as e:   
    return f"ERROR: {str(e)}"