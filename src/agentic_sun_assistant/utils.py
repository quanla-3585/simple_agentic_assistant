import logging

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from domain.tools.tools import *
from domain.tools.utils import *


def parse_message(message: BaseMessage) -> dict:
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
