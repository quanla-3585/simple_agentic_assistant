from langgraph.graph import MessagesState
from typing import Optional

class GenericState(MessagesState):
    user: str
    router_output: Optional[str] = None