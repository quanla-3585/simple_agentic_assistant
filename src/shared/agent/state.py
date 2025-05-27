from typing import Optional
from typing import List
from typing import Annotated
from typing_extensions import TypedDict
from enum import Enum

from langgraph.graph import MessagesState
from langchain_core.messages.tool import ToolCall

def _combine_list(list1, list2):
    return list1 + list2

class QuestionType(Enum):
    """Either RAG (info within Sun*'s documents) or tavily_search"""
    WEB_EXTERNAL = "web_external"
    SUN_INTERNAL = "sun_internal"
    MISC_INFO    = "miscelancenous_informations"

class Question(TypedDict):
    """One question that must be retrieved from outside of LLM"""
    question_type: QuestionType
    question     : str
    sub_questions: Optional[list] = None

class InitialPlan(TypedDict):
    """An ordered list of questions that must be asked"""
    plan_full_text: str
    questions     : List[Question]

class MainAgentState(MessagesState):
    """Provides main agent state
    Properties:
       - user_questions  : user questions, can contain follow ups
       - reasoning_traces: reasoning text collected overtime
       - tool_calls_agg  : aggregated tool call for this user_question, naive TCS
       - final_answer    : graph output for this user_question
    """

    user_questions    : List[str]
    initial_plan      : InitialPlan
    reasoning_traces  : List[str] = []
    tool_calls_agg    : Annotated[List[ToolCall], _combine_list]
    final_answer      : str = ""
    len_msgs_last_loop: int = 0