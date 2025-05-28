import logging
import os

from typing import List
from langchain_openai import AzureChatOpenAI

from langgraph.graph import START, END, StateGraph

from domain.tools.tools import *
from domain.tools.utils import *
from shared.settings.main.configuration import Configuration
from shared.agent.state import MainAgentState, InitialPlan


from prompts import SYSTEM_PROMPT, USER_PROMPT
from state import PlannerInput, PlannerOutput, PlannerState
from state import SubQuestion, SubQuestions
# from utils import parse_message

from infra.llm.settings import Settings
from infra.llm.aws_claude_service import AWSClaudeService, AWSClaudeInput

from dotenv import load_dotenv
load_dotenv()


#init llm services
settings = Settings()
# settings.Config.env_file = "../../../.env"
aws_claude_service = AWSClaudeService(settings=settings.aws_claude)

def generate_plan(state: PlannerState) -> PlannerState:
    """
    Generate sub-questions using Claude.
    """
    question = state['question']
    conversation_history = state.get('conversation_history', '')
    

    llm_input = AWSClaudeInput(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT.format(question=question, conversation_history=conversation_history),
        response_model=SubQuestions
    )

    print(llm_input)
    
    # try:
    response = aws_claude_service._inference_by_llm(llm_input)
    # except Exception as e:
    #     print(f"Error generating planner response: {e}")
    #     return PlannerState(question=question, sub_questions=[])
    
    print(response)

    if not response or not response.questions:
        print("No sub-questions generated.")
        return PlannerState(question=question, sub_questions=[])
    
    return PlannerState(
        question=question,
        sub_questions=[{"question": sq.question, "description": sq.description} for sq in response.questions]
    )

def create_and_compile_graph():
    # Build the graph
    main_agent_builder = StateGraph(PlannerState)
    main_agent_builder.add_node("planning", generate_plan)

    main_agent_builder.add_edge(START, "planning")
    main_agent_builder.add_edge("planning", END)
    
    compiled_graph = main_agent_builder.compile()
    # compiled_graph.invoke()

    return compiled_graph

graph = StateGraph(PlannerState, input=PlannerInput, output=PlannerOutput)

# add node 
graph.add_node("generate_planner_response", generate_plan)

# add relationship 
graph.add_edge(START, "generate_planner_response")
graph.add_edge("generate_planner_response", END)

chatbot = graph.compile()

state_input = PlannerInput(question="Tôi mất máy tính công ty cấp thì phải làm sao ?", conversation_history="")

results = chatbot.invoke(state_input)
print(results.keys())


for sub_question in results['sub_questions']:
    print(f"Sub-question: {sub_question['question']}")
    print(f"Description: {sub_question['description']}")