from openai import OpenAI
import streamlit as st
from agentic_sun_assistant.graph import create_and_compile_graph
from pydantic import BaseModel
from typing import List

import dotenv

dotenv.load_dotenv("./.env")

class FinalAnswerView():
    answer: str
    initial_plan: str
    tool_call_sequence: List[str]
    reasoning_traces: List[str]

agent_graph = create_and_compile_graph()

def parse_messages_thread(
        messages_thread
    ) -> FinalAnswerView:
    """Parse a message thread (one question asked) and return to main chatbox"""
    tool_call_sequence = []
    answer = ""
    reasoning_trace_full_text = ""
    initial_plan = ""
    for msg in messages_thread:
        # Pass human messages
        if msg.type == "user" or msg.type == "human":
            continue
        elif msg.type == 'ai' or msg.type == "assistant":
            # If we found tool calls
            if msg.tool_calls:
                for t_ in msg.tool_calls:
                    tool_name = t_["name"]
                    tool_call_sequence.append(tool_name)
                    if "Planner" == tool_name:
                        initial_plan=t_["args"]["full_plan_text"]
            if "[ANSWER]" in msg.content:
                answer = msg.content
            if "[REASONING]" in msg.content:
                reasoning_trace_full_text += msg.content
    
    reasoning_trace_formatted = "\n🗣🗣🗣".join(
            reasoning_trace_full_text.split("[ANSWER]")[0].split("[REASONING]")
        ).replace("\n\n", "\n")

    print(reasoning_trace_formatted, reasoning_trace_full_text, answer, tool_call_sequence)

    return answer, initial_plan, tool_call_sequence, reasoning_trace_formatted

#################### SCAFFOLDING APP FOR SINGLE-TURN CONVERSATIONS ####################
with st.sidebar:
    "SIDEBAR. UNDER CONSTRUCTION"

st.title("💬 Chatbot")
st.caption("🚀 Agentic SunBot powered by Azure OpenAI, built with sheer willpower")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

import asyncio

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    # client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # print(st.chat_message("user").)
    # st.session_state.messages = agent_graph.get_state()["messages"]
    response = asyncio.run(
        agent_graph.ainvoke({"messages":st.session_state.messages})
    )

    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    st.session_state.messages += response["messages"]
    parsed_result = parse_messages_thread(response["messages"])
    
    msg = response["messages"][-1].content

    #clean up    
    if "[REASONING]" in msg and "[ANSWER]" in msg:
        msg = msg.split("[ANSWER]")[-1]
    msg = msg.replace("[ANSWER]", '')
    
    st.chat_message("assistant").write(msg)
    st.divider()
    st.text(f"🔧 TOOL CALL SEQUENCE : {parsed_result[2]}")
    # st.text(f"🗺 INITIAL PLAN       : {parsed_result[1]}")
    st.text(f"🤔 REASONING TRACES   :\n{parsed_result[3]}")