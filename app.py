from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from pandasai.connectors import MySQLConnector
from pandasai import SmartDataframe
import streamlit as st
from PIL import Image 
import pyodbc
import glob
import os
from voice_recognition import record_and_recognize

# Load environment variables from a .env file

load_dotenv()


# Initialize the model

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

#llm = Ollama(model="llama3")




# Initialize the database connection using pyodbc for SQL Server LocalDB and Windows Authentication

def init_database(
        db_type: str = "",
        user: str = None, 
        password: str = None, 
        host: str = "", 
        port: str = "", 
        mysql_database: str = "database", 
        server: str = "", 
        sqlserver_database: str = "",
        driver: str = ""
        ) -> SQLDatabase:
    """
    Initialize a database connection with the given parameters.
    
    Args:
        db_type (str): Type of database ('mysql' or 'sqlserver').
        user (str): Username for the database.
        password (str): Password for the database.
        host (str): Host address of the database.
        port (str): Port of the database.
        database (str): Database name.
        
    
    Returns:
        SQLDatabase: An SQLDatabase object connected to the database.
    """
    if db_type.lower() == "mysql":
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{mysql_database}"
    elif db_type.lower() == "sqlserver":
        db_uri = f"mssql+pyodbc://@{server}/{sqlserver_database}?driver={driver}"
    else:
        raise ValueError("Unsupported database type. Use 'mysql' or 'sqlserver'.")
    
    return SQLDatabase.from_uri(db_uri)






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
  
  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
  
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
  

  llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
  
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
        return f"Sorry, I couldn't find any data related to your question. Please try asking something else."
    


# Check if a table exists in the database

def table_exists(db, table_name):
    return table_name in db.get_table_names()


# Initialize the MySQL connection using MySQLConnector from pandasai

def init_MySQLConnector_pandasai(user: str, password: str, host: str, port: str, database: str, table: str) -> MySQLConnector:
    return MySQLConnector(
        config={
            "host": host,  
            "port": int(port),  
            "database": database,  
            "username": user,  
            "password": password,  
            "table": table  
        }
    )


# Initialize chat history if not already present

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello 👋! How can I assist you?"),
    ]


# Set up the Streamlit page

st.set_page_config(page_title="Chat With Your Database", page_icon=":speech_balloon:")

st.title("Chat with Your Database")

# Add voice recognition JavaScript



# Sidebar for database connection settings

with st.sidebar:
    #st.image("https://1000logos.net/wp-content/uploads/2017/06/Vodafone_Logo.png")
    st.subheader("Settings")
    st.write("Connect To Your local database and start chatting.")
    
    db_type = st.selectbox("database Type", ["MySQL", "SQLServer"], key="DB_Type")

    if st.session_state["DB_Type"] == "MySQL":
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="Ab59678918", key="Password")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("Database", value="chinook", key="MySQL_Database")
    else:
        st.text_input("Server", value="BASSIONY", key="Server")
        st.text_input("database", value="Chatbot_DB", key="SQLServer_Database")
        st.selectbox("Driver", ["ODBC+Driver+17+for+SQL+Server", "ODBC+Driver+18+for+SQL+Server"], key="Driver")



    if st.button("Connect"):
      with st.spinner("Connecting to database..."):
          try:
              if st.session_state["DB_Type"] == "MySQL":
                db = init_database(
                    st.session_state["DB_Type"],
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["MySQL_Database"],
                )

              else:
                  db = init_database(
                    st.session_state["DB_Type"],
                    st.session_state["Server"],
                    st.session_state["SQLServer_Database"],
                    st.session_state["Driver"],
                )

              st.session_state.db = db
              st.success("Connected to database!")
          except Exception as e:
              st.error(f"Connection failed: {e}")

# Display chat history

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="F:\Downloads\downloads\Vodafone-Chatbot-main (1)\Vodafone-Chatbot-main\src\sql.png"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)


# Initialize session state for the recognized text
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""  # Initialize with an empty string

# Handle user input
col1, col2 = st.columns([4, 1])

with col1:
    # Use a regular text input instead of st.chat_input
    user_query = st.text_input("Type a message...", value=st.session_state.recognized_text)

with col2:
    if st.button("Press to Record", key="record_button"):
        result = record_and_recognize()
        if result.startswith("Recognized Text:"):
            # Update the session state with the recognized text
            recognized_text = result.replace("Recognized Text:", "").strip()
            st.session_state.recognized_text = recognized_text
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()



#database = st.session_state["MySQL_Database"]
# Define the temporary folder path
temp_folder = r"C:\Users\besho\AppData\Local\Temp"

if "db" not in st.session_state:
    st.session_state["db"] = None

# Your existing code
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI", avatar="F:\Downloads\downloads\Vodafone-Chatbot-main (1)\Vodafone-Chatbot-main\src\sql.png"):
        
        check_plotting_chain = check_plotting()
        st.session_state.boolean_plotting = check_plotting_chain.invoke({
            "question": user_query
        })

        if st.session_state.boolean_plotting == "True":
        
            sql_chain = get_sql_table(st.session_state.db)
            st.session_state.table_name = sql_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "question": user_query
            })
            
            if table_exists(st.session_state.db, st.session_state.table_name):

                st.session_state.my_connector = init_MySQLConnector_pandasai(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["MySQL_Database"],
                    st.session_state["table_name"],
                )

                # Initialize the SmartDataframe with the updated connector

                df_connector = SmartDataframe(st.session_state.my_connector, config={"llm": llm})

                response = df_connector.chat(user_query)

                image_path = r'C:\Users\moham\Desktop\Projects\Vodafone-Chatbot\src\exports\charts\temp_chart.png'
                # Open the image using PIL
                #image = Image.open(image_path)

                # Display the image in Streamlit
                #st.image(image)

                # #st.markdown(st.image(image))
                st.session_state.chat_history.append(AIMessage(content=response))

                with open(image_path, "rb") as file:
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=image_path,
                        mime="image/png"
                    )




            else:
                fallback_response = "I'm sorry, but I couldn't find the information you're looking for in the database."
                st.markdown(fallback_response)
                st.session_state.chat_history.append(AIMessage(content=fallback_response))

        else:

            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)

            st.session_state.chat_history.append(AIMessage(content=response))
