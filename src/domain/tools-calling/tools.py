
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

from langchain_core.tools import tool

from shared.settings.main.configuration import Configuration
from shared.agent.state import *

from .utils import *

class DepartmentEnum(str, Enum):
    rnd = 'RND'
    aie = 'AIE'
    ifu = 'IFU'

@tool(
    name_or_callable="get_time_now",
    description="""
    Return full system time of right now.
    """
)
def get_time_now() -> str:
    from datetime import datetime
    return str(datetime.now())

@tool(
    name_or_callable="RAG",
    description="""
    Use this tool to query for information only for internal org's documents from Sun Asterisk.
    Currently, information is sparse and ONLY documents pertaining internal process and audit for handling documents is present. 
    Any information not related to Sun Asterisk is not here.
    The tool will return relevant context that can be used to answer the user's question.
    """
)
class RAG(BaseModel):
    query: str = Field(
        description="The search query or question to retrieve information for.",
    )
    department: DepartmentEnum = Field(
        description="Department that is in charge of this information domain.",
    )
    max_results: Optional[int] = Field(
        description="Maximum number of results to return. Default is 3.",
        default=3
    )

@tool(
    name_or_callable="SimpleQAAgentHandover",
    description="""
    Handover a question to a simple QA agent for processing. This tool allows the main agent to delegate 
    straightforward questions to a specialized QA agent that can provide direct answers without complex reasoning.
    Use this tool when the question is factual, direct, and doesn't require complex reasoning or multi-step processing.
    """
)
class SimpleQAAgentHandover(BaseModel):
    question: str = Field(
        description="The question to be handed over to the QA agent.",
    )
    context: Optional[str] = Field(
        description="Optional context that might help the QA agent answer the question more accurately.",
        default=None
    )
    max_tokens: Optional[int] = Field(
        description="Optional maximum number of tokens for the response. Default is 500.",
        default=500
    )
    main_department: DepartmentEnum = Field(
        description="Department that is in charge of answering this question.",
    )

@tool(
    name_or_callable="QuestionSet", 
    description="""
    Make up a list of cohesive, comprehensive questions regarding the user's input. The question must reflect the user's intentions. These will later be used for searching.
    Use the QuestionSet tool to create the questions set, the QuestionSet tool have only 2 fields:
      -"queries": containing statements/expressions/questions as strings for later precise information retrieval.
      -"departments": List of Departments that is in charge of answering this question. Chosen only from the list of: RND, HR, SWE, Infra, CnB
""")
class QuestionSet(BaseModel):
    queries: List[str] = Field(
        description="Questions generated based on decomposing the user input.",
    )
    departments: List[str] = Field(
        description="Department that is in charge of answering this question. Chosen only from the list of: RND, HR, SWE, Infra, CnB, Sec",
    )

@tool(
    name_or_callable="Rephraser", 
    description="""
    Ensure that the question user asked is linguistically compatible with the provided context.
    Any convoluted, confusing in phrasing, language choice or intonations must be rephrased into a cohesive, coherent question.
    Do this on a best-effort mindset.
""")
class Rephraser(BaseModel):
    rephrased_question: str = Field(
        description="The question, cleaned up and rephrased."
    )

@tool(
    name_or_callable="Planner", 
    description="""
    ALWAYS use this everytime the user ask anything. 
    This is not user's planner, this is assistant's planner tool.
    Sunbot's personal planner for drafting plans, only have 1 field:
      - full_plan_text: the plan of execution in full. A plan is a list containing multiple steps of either reasoning or calling tools at each step.
""")
class Planner(BaseModel):
    full_plan_text: str = Field(
        description="The plan. Simply a list of steps that must be taken. In string."
    )

@tool(
    name_or_callable="get_org_departments",
    description="Get the general departments within the organization and a short description of their suite of documents"
)
def get_org_departments():
    
    return {
        "Sec":"Data security, defensive techniques, sec routines, systems design checklist",
        "CnB":"Compensation, salaries, bonuses, insurances, holiday pay, PTOs. Everything money included",
        "HR":"Human resources, Company policy, internal affairs, Equipments handling, Equipments usage policies",
        "CnE":"Engineering, everything technical, internal tools user manuals, internal tools dev manual"
    }