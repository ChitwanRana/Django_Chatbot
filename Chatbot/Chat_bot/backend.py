from dotenv import load_dotenv
import os
import sqlite3
import requests
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
        ops = {
            "+": lambda a,b: a+b,
            "add": lambda a,b: a+b,
            "-": lambda a,b: a-b,
            "sub": lambda a,b: a-b,
            "*": lambda a,b: a*b,
            "mul": lambda a,b: a*b,
            "/": lambda a,b: a/b if b != 0 else "DivisionByZero",
            "div": lambda a,b: a/b if b != 0 else "DivisionByZero",
        }
        op = operation.strip().lower()
        if op in ops:
            return {"result": ops[op](first_num, second_num)}
        return {"error": f"Unsupported operation '{operation}'"}
    except Exception as e:
        return {"error": str(e)}

@tool("Stock_Price")
def get_stock_price(symbol: str) -> dict:
    """Stock Price Retriever that fetches Stock Data."""
    av_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not av_key:
        return {"error": "ALPHAVANTAGE_API_KEY not configured"}
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={av_key}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# ============= STATE & GRAPH =============
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_context: str

def retrieve_node(state: ChatState):
    query = state.get("query", "")
    context = ""
    try:
        if os.path.isdir(VECTOR_STORE_PATH) and any(os.scandir(os.path.dirname(VECTOR_STORE_PATH))):
            vectorstore = load_vectorstore()
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = ""
    except Exception as e:
        context = ""
    return {"retrieved_context": context}

def chat_node(state: ChatState):
    messages = state.get("messages", [])
    context = state.get("retrieved_context", "")
    # Safe-get last user message (if any)
    user_msg = ""
    if messages:
        last = messages[-1]
        user_msg = getattr(last, "content", "") or ""
    system_prompt = f"Use context below to answer accurately:\n\n{context}\n\nQuestion: {user_msg}"
    # Invoke LLM with only a system prompt followed by the recent messages to avoid echoing the question twice
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
    Process user input (non-echoing) and yield a single assistant response.
    Safely extract text whether the chatbot returns dicts or message objects.
    """
    from langchain_core.messages import HumanMessage
    CONFIG = {"configurable": {"thread_id": thread_id}}
    try:
        result = chatbot.invoke({"messages": [HumanMessage(content=user_input)], "query": user_input}, config=CONFIG)
        assistant_text = ""

        # If result is a dict-like structure
        if isinstance(result, dict):
            msgs = result.get("messages") or []
            if msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    assistant_text = last.get("content") or last.get("text") or ""
                elif hasattr(last, "content"):
                    assistant_text = getattr(last, "content", "") or ""
                elif hasattr(last, "text"):
                    assistant_text = getattr(last, "text", "") or ""
                else:
                    assistant_text = str(last)
            else:
                assistant_text = result.get("output", "") or str(result)

        else:
            # result may be an object with .messages or .content attributes
            if hasattr(result, "messages"):
                msgs = getattr(result, "messages") or []
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "content"):
                        assistant_text = getattr(last, "content", "") or ""
                    elif hasattr(last, "text"):
                        assistant_text = getattr(last, "text", "") or ""
                    else:
                        assistant_text = str(last)
                else:
                    assistant_text = str(result)
            else:
                assistant_text = getattr(result, "content", getattr(result, "text", str(result)))

    except Exception as e:
        assistant_text = f"Error: {e}"

    yield assistant_text

def get_thread_history(thread_id):
    """Retrieve message history for a given thread from the checkpointer."""
    history = []
    try:
        cfg = {"configurable": {"thread_id": thread_id}}
        thread_state = checkpointer.get(cfg)
        if not thread_state:
            return history
        # thread_state may contain a 'values' dict with messages
        values = thread_state.get("values", thread_state) if isinstance(thread_state, dict) else thread_state
        msgs = values.get("messages", []) if isinstance(values, dict) else []
        for m in msgs:
            # m might be an object with .content or a dict
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if not role and hasattr(m, "__class__"):
                clsname = m.__class__.__name__.lower()
                if "human" in clsname:
                    role = "user"
                elif "ai" in clsname or "assistant" in clsname:
                    role = "assistant"
            if content:
                history.append({"role": role if role in ("user", "assistant") else "assistant", "content": content})
    except Exception:
        pass
    return history

def delete_thread_history(thread_id):
    """Deletes a thread's history from the checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpointer.put(config, None) # Deleting by saving None

def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            try:
                cfg = checkpoint.get("configurable") if isinstance(checkpoint, dict) else None
                if isinstance(cfg, dict):
                    tid = cfg.get("thread_id") or cfg.get("id")
                    if tid:
                        all_threads.add(tid)
            except Exception:
                continue
    except Exception:
        pass
    return list(all_threads)
