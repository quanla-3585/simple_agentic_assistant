from typing import Optional, List, Annotated

from langgraph.graph import MessagesState
from langchain_core.messages.tool import ToolCall

def _combine_list(list1, list2):
    return list1 + list2
def _replace(a, b):
    return b


class MainAgentState(MessagesState):
    """Provides main agent state
    Properties:
       - user_questions  : user questions, can contain follow ups
       - reasoning_traces: reasoning text collected overtime
       - tool_calls_agg  : aggregated tool call for this user_question, naive TCS
       - final_answer    : graph output for this user_question
    """

    user_questions    : List[str]
    reasoning_traces  : List[str] = []
    tool_calls_agg    : Annotated[List[ToolCall], _combine_list]
    final_answer      : str = ""
    len_msgs_last_loop: int = 0