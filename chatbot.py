import os 
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from tools import query_knowledge_base, search_for_product_recommendations

llama_llm = ChatOpenAI(
    api_key="ollama",
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
)

prompt = """#Purpose 

You are a customer service chatbot for a flower shop company. You can help the customer achieve the goals listed below.

#Goals

1. Answer questions the user might have relating to serivces offered
2. Recommend products to the user based on their preferences

#Tone

Helpful and friendly. Use gen-z emojis to keep things lighthearted. You MUST always include a funny flower related pun in every response."""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{messages}")
    ]
)

tools = [query_knowledge_base, search_for_product_recommendations]

llm_with_tools = chat_template | llama_llm.bind_tools(tools)
# llm = chat_template | llama_llm

def call_agent(message_state: MessagesState):
    response = llm_with_tools.invoke(message_state)

    return { "messages" : [response]}

def is_there_tool_calls(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"
    
# create ReAct graph

builder = StateGraph(MessagesState)
builder.add_node("agent", call_agent)
builder.add_node("tool_node", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", is_there_tool_calls)
builder.add_edge("tool_node", "agent")


app = builder.compile()