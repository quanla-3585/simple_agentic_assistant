import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_ANSWER_STRUCTURE = """Use this structure to create an answer on the user-provided question:

1. Condensed Answer
   - The answer for the user's question, as concise and precise as possible
2. Relevant Information:
   - Elaborate your Condensed Answer if needed.

"""

class SearchAPI(Enum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    # Common configuration
    report_structure: str = DEFAULT_ANSWER_STRUCTURE # Defaults to the default report structure

    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    
    # Graph-specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations

    # Question Agent
    question_agent_provider: str = "openai"
    question_agent_model: str = "gpt-4.1-nano"
    question_agent_model_kwargs: Optional[Dict[str, Any]] = None

    # Answer Agent
    main_agent_provider: str = "openai"
    main_agent_model: str = "gpt-4.1-nano"
    main_agent_model_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
