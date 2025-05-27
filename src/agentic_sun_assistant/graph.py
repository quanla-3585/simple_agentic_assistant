import logging
import os

from typing import List, Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from langgraph.types import Command
from langgraph.graph import START, END, StateGraph

from domain.tools.tools import *
from domain.tools.utils import *
from shared.settings.main.configuration import Configuration
from shared.agent.state import MainAgentState, InitialPlan
from shared.mockups.rag_db import MOCK_KNOWLEDGE_BASE


async def tools_executor_agent(state: MainAgentState, config: Configuration):
    """Agent forced to do tool calling"""
    
    # Get all tools
    available_tools, _ = get_main_agent_tools(config)

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

    # Always planning first step
    if len(messages)==1:
        tool_choice="Planner"

    llm_response = await llm.bind_tools(available_tools, tool_choice="any", parallel_tool_calls=True).ainvoke(
                [
                    {
                        "role": "system",
                        "content": """
                        Based on the provided input from the converstation history.
                        Generate a set of tool calls that satisfy the questions planned to ask.
                        """
                    }
                ]
                + messages
            )

    # some cheap trick to init state, TODO: refactor
    if len(messages) == 1:
        state.update({"reasoning_traces": []})
        state.update({"tool_calls_agg"  : []})
        state.update({"final_answer"    : ""})
        state.update({"user_questions"  : []})
    else:
        # for i_ in range(state["len_msgs_last_loop"], len(llm_response)+len(messages)):
        llm_response_parsing = parse_message(llm_response)
        try:
            for key_ in llm_response_parsing.keys():
                if key_ == "messages": continue
                if isinstance(state[key_], List): 
                    state.update({key_: state[key_]+llm_response_parsing[key_]})
                else: state.update({key_: llm_response_parsing[key_]})
        except Exception as e:
            logging.warning(f"STATE UPDATE FAILED with parsed info {llm_response_parsing}, threw exception: {e}")

    #propagate the message stack
    state.update({"len_msgs_last_loop": len(messages)})
    
    #update some states
    state.update({"messages":llm_response})

    return state

async def traces_parser(state: MainAgentState, config: Configuration):
    """For logging"""
    
    messages = state["messages"]

    for message in messages:
        # for i_ in range(state["len_msgs_last_loop"], len(llm_response)+len(messages)):
        llm_response_parsing = parse_message(message)
        try:
            for key_ in llm_response_parsing.keys():
                if key_ == "messages": continue
                if isinstance(state[key_], List): 
                    state.update({key_: state[key_]+llm_response_parsing[key_]})
                else: state.update({key_: llm_response_parsing[key_]})
        except Exception as e:
            logging.warning(f"STATE UPDATE FAILED with parsed info {llm_response_parsing}, threw exception: {e}")
    # #update some states
    #     state.update({"messages":llm_response})

    return state

async def answer_generator(state: MainAgentState, config: Configuration):
    # Load sensitive config from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    model = AzureChatOpenAI(
        azure_endpoint="https://sun-ai.openai.azure.com/",
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
    )

    # Get current messages stack. This is simple message caching
    messages      = state["messages"]

    # assert isinstance(human_message, HumanMessage), "Handling AI message is currently unavailable"
    response  = await model.ainvoke(
        [
            {
                "role": "system",
                "content": ANSWERER_PROMPT
            }
        ]
        + messages
    )
    state.update({"final_answer":response.content})
    
    return state

def invoke_pseudo_rag(args):
    """
    Mock implementation of a RAG system that retrieves information from a simulated knowledge base.
    """
    query = args.get("query", "").lower()
    department = args.get("department")
    max_results = args.get("max_results", 3)
    
    # Get the department data
    dept_data = MOCK_KNOWLEDGE_BASE.get(args.get("department", "Common"), MOCK_KNOWLEDGE_BASE["Common"])
    
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
    
    # return results
    # return "\n\n".join(results)
    return MOCK_KNOWLEDGE_BASE

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
    tool_list = [get_time_now, RAG, websearch_tool]
    return tool_list, {tool.name: tool for tool in tool_list}

async def main_agent_tools(
        state: MainAgentState, config: Configuration
    ) -> Command[Literal["answer_generator"]]:
    
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
    return Command(goto="answer_generator", update={"messages": result})

async def initial_planning(state: MainAgentState, config: Configuration):

    # Load sensitive config from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    model = AzureChatOpenAI(
        azure_endpoint="https://sun-ai.openai.azure.com/",
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
    )

    # Get current messages stack. This is simple message caching
    messages      = state["messages"]
    human_message = messages[-1]
    # assert isinstance(human_message, HumanMessage), "Handling AI message is currently unavailable"

    model_with_structure = model.with_structured_output(InitialPlan)
    structured_response  = await model_with_structure.ainvoke(
        [
            {
                "role": "system",
                "content": PLANNER_PROMPT
            }
        ]
        + [human_message]
    )
    state.update({"initial_plan":structured_response})
    
    return state

def create_and_compile_graph():
    # Build the graph
    main_agent_builder = StateGraph(MainAgentState, config_schema=Configuration)
    main_agent_builder.add_node("tools_executor_agent", tools_executor_agent)
    main_agent_builder.add_node("main_agent_tools", main_agent_tools)
    main_agent_builder.add_node("initial_planning", initial_planning)
    main_agent_builder.add_node("answer_generator", answer_generator)
    main_agent_builder.add_node("traces_parser", traces_parser)

    main_agent_builder.add_edge(START, "initial_planning")
    main_agent_builder.add_edge("initial_planning", "tools_executor_agent")
    main_agent_builder.add_edge("tools_executor_agent", "main_agent_tools")
    main_agent_builder.add_edge("main_agent_tools", "answer_generator")
    main_agent_builder.add_edge("answer_generator", "traces_parser")
    main_agent_builder.add_edge("traces_parser", END)
    
    compiled_graph = main_agent_builder.compile()
    # compiled_graph.invoke()

    return compiled_graph

graph = create_and_compile_graph()