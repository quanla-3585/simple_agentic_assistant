from typing import TypedDict, Optional
from pydantic import BaseModel, Field

# define state graph
class PlannerInput(TypedDict):
    question: str
    conversation_history: str

class PlannerOutput(TypedDict):
    sub_questions: list 

class PlannerState(TypedDict):
    question: str
    sub_questions: Optional[list] = None
    conversation_history: str

# datamodels for LLM structured outputs
class SubQuestion(BaseModel):
    question: str = Field(..., description="The sub-question generated from the main question.")
    description: str = Field(..., description="A brief description explaining the purpose of the sub-question.")

class SubQuestions(BaseModel):
    questions: list[SubQuestion] = Field(
        ...,
        description="A list of sub-questions generated from the main question, each with a description."
    )
