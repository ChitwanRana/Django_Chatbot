from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os, sqlite3, requests

dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=azure_deployment,
    api_version=azure_api_version,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_embedding_deployment, 
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)

# ============= VECTOR STORE =============
VECTOR_STORE_PATH = "vectorstore/faiss_index"

if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

def load_vectorstore():
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# ============= TOOLS =============
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool("Calculator")
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Calculator that performs basic arithmetic operations."""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero!"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool("Stock_Price")
def get_stock_price(symbol: str) -> dict:
    """Stock Price Retriever that fetches Stock Data."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# ============= STATE & GRAPH =============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_context: str

def retrieve_node(state: ChatState):
    query = state["query"]
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = load_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
    else:
        context = "(No document uploaded yet)"
    return {"retrieved_context": context}

def chat_node(state: ChatState):
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    user_msg = messages[-1].content
    system_prompt = f"Use context below to answer accurately:\n\n{context}\n\nQuestion: {user_msg}"
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ============= UTILS =============
def process_user_query(thread_id, user_input):
    """
    Process user input as a stream and yield AI response chunks.
    """
    from langchain_core.messages import HumanMessage
    CONFIG = {"configurable": {"thread_id": thread_id}}
    
    # The event stream format is "event: <event_name>\ndata: <data>\n\n"
    # For simplicy, we'll just yield the data chunks.
    for chunk in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)], "query": user_input},
        config=CONFIG,
        stream_mode="values", # Use "values" to get the content of the nodes
    ):
        # We are interested in the output of the 'chat_node'
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            if last_message.content:
                yield last_message.content

def get_thread_history(thread_id):
    """Retrieve message history for a given thread."""
    from langchain_core.messages import AIMessage, HumanMessage
    
    history = []
    thread_state = checkpointer.get({"configurable": {"thread_id": thread_id}})
    
    if thread_state:
        for msg in thread_state["values"]["messages"]:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})
            
    return history

def delete_thread_history(thread_id):
    """Deletes a thread's history from the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer.put(config, None) # Deleting by saving None

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
