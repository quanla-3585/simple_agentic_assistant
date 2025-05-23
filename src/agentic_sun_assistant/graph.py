from typing import List, Annotated, TypedDict, operator, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

from agentic_sun_assistant.configuration import Configuration
from agentic_sun_assistant.utils import *
from agentic_sun_assistant.prompts import *
from agentic_sun_assistant.state import MainAgentState
from agentic_sun_assistant.tools import *
from agentic_sun_assistant.rag_db import MAIN_CHUNKS

import logging
import os

def _parse_message(message: BaseMessage) -> dict:
    """Parse individual message to my state defs, ready to merge to state
        In:
            message: the message
        Out:
            results: a dict containing fields ready to merge/append/override my current state
    """
    results = {}
    if isinstance(message, HumanMessage):
        return {"user_questions": [message.content], "final_answer": ""} #refresh final answer at every new follow ups
    
    elif isinstance(message, AIMessage):
        logging.info(f"PARSING AI MESSAGE: {message}")
        if message.tool_calls:
            results["tool_calls_agg"] = [message.tool_calls]
        if "[ANSWER]" in message.content:
            try:
                results["final_answer"] =  message.content.split("[ANSWER]")[-1]
            except:
                logging.warning("Failed Answer parsing, dumping all to var.")
                results["final_answer"] =  message.content
        if "[REASONING]" in message.content:
            results["reasoning_traces"] = [message.content]
    
    else:
        logging.warning(f"Unknown message instance passed for parsing {message.type}")

    return results
    
        
async def main_agent(state:MainAgentState, config: Configuration):
    
    """Agent to make more questions"""
    
    # Get all tools
    available_tools, _ = get_main_agent_tools(config)

    # DEPRECATED: Get configuration, Initialize the model
#     configurable = Configuration.from_runnable_config(config)
#     main_agent_model = get_config_value(configurable.main_agent_model)
#     llm = init_chat_model(model=main_agent_model, temperature=0.2)

    # Load sensitive config from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    llm = AzureChatOpenAI(
        azure_endpoint="https://sun-ai.openai.azure.com/",
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
    )

    # Get current messages stack. This is simple message caching
    messages = state["messages"]
    
    # Incase there are failures
    llm_response = messages
    llm_response = await llm.bind_tools(available_tools, tool_choice="auto").ainvoke(
                [
                    {
                        "role": "system",
                        "content": QUESTION_SYNTHESIZING_INSTRUCTION
                    }
                ]
                + messages
            )

    # some cheap trick to init state, TODO: refactor
    if len(messages) == 1:
        state.update({"reasoning_traces": []})
        state.update({"tool_calls_agg"  : []})
        state.update({"final_answer"    : ""})
    else:
        # for i_ in range(state["len_msgs_last_loop"], len(llm_response)+len(messages)):
        llm_response_parsing = _parse_message(llm_response)
        try:
            for key_ in llm_response_parsing.keys():
                if key_ == "messages": continue
                if isinstance(state[key_], list): 
                    state.update({key_: state[key_]+llm_response_parsing[key_]})
                else: state.update({key_: llm_response_parsing[key_]})
        except Exception as e:
            logging.warning(f"STATE UPDATE FAILED with parsed info {llm_response_parsing}, threw exception: {e}")

    #propagate the message stack
    state.update({"len_msgs_last_loop": len(messages)})
    
    #update some states
    state.update({"messages":llm_response})

    return state

async def simple_qa_agent(state:MessagesState, config: Configuration):
    """
    Simple QA agent that provides direct answers to straightforward questions.
    This agent is designed to handle factual, direct questions without complex reasoning.
    """
    
    # Get configuration
    # configurable = Configuration.from_runnable_config(config)
    # qa_agent_model = get_config_value(configurable.question_agent_model)
    
    # Initialize the model
    # llm = init_chat_model(model=qa_agent_model)
    # Load sensitive config from environment variables

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    llm = AzureChatOpenAI(
        azure_endpoint="https://sun-ai.openai.azure.com/",
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
        )

    # Extract the handover data from the last message
    # Prepare the context for the QA agent
    
    # Get the answer from the model
    response = llm.invoke([
        {"role": "system", "content": SIMPLE_QA_AGENT_INSTRUCTION}] + state["messages"]
    )
    
    # Return the answer
    return {"messages": [{"role": "assistant", "content": response.content}]}

# Mock database for RAG
MOCK_KNOWLEDGE_BASE = {
    "Common": MAIN_CHUNKS,
    "RND": {
        "ai": [
            "AI research at our company focuses on large language models and their applications in business contexts.",
            "Our R&D team has developed several prototype AI systems for internal use, including document analysis and code generation.",
            "The latest AI project involves developing a RAG system that can query our internal knowledge graph."
        ],
        "research": [
            "Research methodologies employed by our team include A/B testing, user studies, and literature reviews.",
            "Our research process typically involves hypothesis formation, experimentation, and peer review.",
            "Recent research has focused on improving retrieval mechanisms for knowledge graphs."
        ],
        "development": [
            "Development practices include agile methodologies, CI/CD pipelines, and code reviews.",
            "Our development team uses Python, TypeScript, and Rust for most projects.",
            "The development roadmap for Q3 includes implementing a new graph database system."
        ]
    },
    "AIE": {
        "engineering": [
            "AI Engineering team is responsible for deploying and maintaining AI systems in production.",
            "Our engineering stack includes Docker, Kubernetes, and various cloud services.",
            "Engineering challenges include scaling inference for large language models."
        ],
        "deployment": [
            "Deployment processes include rigorous testing, gradual rollout, and monitoring.",
            "We deploy new models on a bi-weekly basis after thorough validation.",
            "Deployment metrics include latency, throughput, and error rates."
        ]
    },
    "IFU": {
        "infrastructure": [
            "Infrastructure team manages our cloud resources, on-premise servers, and networking.",
            "Our infrastructure is primarily based on AWS with some services on Azure.",
            "Infrastructure costs are monitored and optimized on a monthly basis."
        ],
        "security": [
            "Security protocols include regular audits, penetration testing, and access control reviews.",
            "All data is encrypted both in transit and at rest using industry-standard protocols.",
            "Security training is mandatory for all employees on a quarterly basis."
        ]
    },
    "HR": {
        "compensation and benefits": [
            "Pay day is at 30th every month."
        ]
    }
}

def invoke_pseudo_rag(args):
    """
    Mock implementation of a RAG system that retrieves information from a simulated knowledge base.
    """
    query = args.get("query", "").lower()
    department = args.get("department")
    max_results = args.get("max_results", 3)
    
    # Get the department data
    dept_data = MOCK_KNOWLEDGE_BASE
    
    # Find relevant information based on the query
    results = []
    for topic, entries in dept_data.items():
        if topic in query or any(query in entry.lower() for entry in entries):
            results.extend(entries)
    
    # If no direct matches, return some general information from the department
    if not results and dept_data:
        for entries in dept_data.values():
            results.extend(entries)
    
    # Limit the number of results
    results = results[:max_results]
    
    if not results:
        return f"No information found for query '{query}' in department {department}."
    
    return MOCK_KNOWLEDGE_BASE
    # return "\n\n".join(results)

def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    if search_api.lower() == "tavily":
        return tavily_search
    else:
        raise NotImplementedError(
            f"Currently, only Tavily is supported."
        )

def get_main_agent_tools(config:Configuration):
    """Get main agent's tools based on configuration"""
    websearch_tool = get_search_tool(config)
    tool_list = [Planner, get_time_now, RAG, websearch_tool]
    return tool_list, {tool.name: tool for tool in tool_list}

async def main_agent_tools(
        state: MainAgentState, config: Configuration
    ) -> Command[Literal["main_agent", "__end__"]]:
    
    result = []
    _, main_agent_tools_by_name = get_main_agent_tools(config)

    # Check last message for tool calls
    for tool_call in state["messages"][-1].tool_calls:
        
        # Get the tool
        tool = main_agent_tools_by_name[tool_call["name"]]
        
        # switch-case for RAG
        if tool_call["name"] == "RAG":
            observation = invoke_pseudo_rag(tool_call["args"])

        # other tool call - use ainvoke for async tools
        elif hasattr(tool, 'ainvoke'):
            observation = await tool.ainvoke(tool_call["args"])
        else:
            observation = tool.invoke(tool_call["args"])

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
    
    # last message no tools
    return Command(goto="main_agent", update={"messages": result})


# async def question_agent_should_continue(state: MainAgentState) -> Literal["main_agent_tools", "simple_qa_agent", END]:
async def question_agent_should_continue(state: MainAgentState) -> Literal["main_agent_tools", "main_agent",  END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    if len(messages)==0:
        return END

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        return "main_agent"

    # If the LLM makes a tool call, then perform an actio
    if last_message.tool_calls:
        return "main_agent_tools"
    # Crude logic to force toolcall
    elif "[REASONING]" in last_message.text() and "[ANSWER]" not in last_message.text():
        return "main_agent"
    else: # This should truncate the ReAct loop
        return END


async def router(state: MessagesState, config: Configuration) -> Literal["main_agent", "simple_qa_agent"]:
    """
    Router function that analyzes the user's question and decides whether to route it to
    the simple_qa_agent or the main_agent.
    
    This is not an agent but a simple router that does some thinking to determine the best path.
    """
    # Get the user's question from the messages
    messages = state["messages"]
    
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    qa_agent_model = get_config_value(configurable.question_agent_model)
    
    # Initialize the model
    # llm = init_chat_model(model=qa_agent_model)
    # Load sensitive config from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    llm = AzureChatOpenAI(
        azure_endpoint="https://sun-ai.openai.azure.com/",
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
    )

    response = await llm.ainvoke([
        {
            "role": "system", 
            "content": """
            
            You are a router for a multi-agents system, do routing for user questions input. There is two agents now in the system for routing: SimpleQA agent or question synthesizer agent.
            Return 'main_agent' if you think this is a complex question that requires more complex handling, return 'simple_qa_agent' if you know that this is a trivial matter and should be handled by some simple agents.
        """},
    ] + messages)

    # print(response)

    return {"router_output":response.content}


def create_and_compile_graph():
    # Build the graph
    main_agent_builder = StateGraph(MainAgentState, config_schema=Configuration)
    main_agent_builder.add_node("main_agent", main_agent)
    main_agent_builder.add_node("main_agent_tools", main_agent_tools)

    main_agent_builder.add_edge(START, "main_agent")
    main_agent_builder.add_conditional_edges(
        "main_agent",
        question_agent_should_continue,
        {
            # Name returned by should_continue : Name of next node to visit
            "main_agent_tools": "main_agent_tools",
            "main_agent": "main_agent",
            END: END,
        },
    )
    main_agent_builder.add_edge("main_agent_tools", "main_agent")
    main_agent_builder.add_edge("main_agent_tools", END)

    # Compile the graph
    graph = main_agent_builder.compile()
    logging.info("main_agent_builder Graph Compiled Successfully !")

    return graph

graph = create_and_compile_graph()