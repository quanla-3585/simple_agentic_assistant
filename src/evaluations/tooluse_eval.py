from agentic_sun_assistant.graph import create_and_compile_graph
from langchain_core.messages import AIMessage
import asyncio
import dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric
import json

DATASET_PATH = "src/evaluations/tools_calling_eval_dataset.json"
eval_dataset = json.load(open(DATASET_PATH, 'r'))

# Real, not mocked
dotenv.load_dotenv()
eval_graph = create_and_compile_graph()

# Build eval set
test_cases = [] 
for case_ in eval_dataset:

    message_trace = asyncio.run(
        eval_graph.ainvoke({"type":"user", "content":case_["user_input"]})
    )

    tools_called = []
    for msg in message_trace["messages"]:
        if isinstance(msg, AIMessage):
            tools_called+=[t["name"] for t in msg.tool_calls]

    test_case = LLMTestCase(
        input=case_["user_input"],
        actual_output=message_trace["messages"][-1].content,
        # Replace this with the tools that was actually used by your LLM agent
        tools_called=[ToolCall(name=n) for n in tools_called],
        expected_tools=[ToolCall(name=n) for n in case_["ideal_tools"]],
    )
    test_cases.append(test_case)

metric = ToolCorrectnessMetric()

# To run metric as a standalone
evaluate(test_cases=test_cases, metrics=[metric])