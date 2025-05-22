import asyncio
import tqdm
import json

import dotenv
import textdistance
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.test_case import ToolCall
from deepeval.metrics   import ToolCorrectnessMetric
from deepeval.metrics   import AnswerRelevancyMetric
from deepeval.evaluate  import DisplayConfig

from langchain_core.messages import AIMessage

from agentic_sun_assistant.graph import create_and_compile_graph


# Real, not mocked
dotenv.load_dotenv()
eval_graph   = create_and_compile_graph()

# DATASET_PATH = "src/evaluations/tools_calling_eval_dataset.json"
DATASET_PATH = "data/tool-calling-eval/TC-R_Eval_cleaned.json"

eval_dataset = json.load(open(DATASET_PATH, 'r')) #[:5]
eval_results = [{} for _ in eval_dataset]

print(eval_graph.input_schema)

async def arun_datapoint(input:str):
    """Run 1 eval datapoint and handle exceptions"""

    try:
        result = await eval_graph.ainvoke(
            {
                "messages":[{"type":"user", "content":input}]
            }
        )
        return result
    except Exception as e:
        print(f"Failed processing input {input}, raised exception {e}")
        return {
            "messages":[]
        }

async def arun_dataset(eval_dataset):
    """async run evaluation on all of the cases"""

    user_inputs    = [_["Question"] for _ in eval_dataset]

    run_results = await tqdm.asyncio.tqdm.gather(
        *[arun_datapoint(ui_) for ui_ in user_inputs]
    )
    
    for i in range(len(user_inputs)):
        eval_results[i]["user_input"]  = user_inputs[i]
        eval_results[i]["ideal_tools"] = eval_dataset[i]["Tools_Accuracy"]["Expected_Tools"]
        eval_results[i]["tcs"]         = eval_dataset[i]["Tool_Call_Sequence_(TCS)"]["Expected_TCS"]
        eval_results[i]["state"]       = run_results[i]

asyncio.run(arun_dataset(eval_dataset))

test_cases   = []
sum_norm_lvs = 0
count_cases  = 0

for run_ in eval_results:

    tools_called = []
    for msg in run_["state"]["messages"]:
        tools_called += getattr(msg, "tool_calls", [])

    tools_called = [_["name"] for _ in tools_called]
    print(tools_called)

    # Collect data for evaluation
    input          = run_["user_input"]
    actual_output  = run_["state"]["messages"][-1].content if len(run_["state"]["messages"])>0 else ""
    tools_called   = [ToolCall(name=n) for n in list(set(tools_called))]
    expected_tools = [ToolCall(name=n) for n in run_["ideal_tools"]] if run_["ideal_tools"] is not None else []

    # Skip cases that does not require a tool chain
    expected_tcs  = run_["tcs"] if run_["tcs"] is not None else []
    if expected_tcs is not []:
        tcs           = tools_called
        max_len       = max(len(expected_tcs), len(tcs))
        lvs_dist      = textdistance.levenshtein.distance(tcs,expected_tcs)
        if max_len == 0:
            norm_lvs_dist = 1    
        else: norm_lvs_dist = lvs_dist/max_len
        sum_norm_lvs += norm_lvs_dist
        count_cases  += 1

    test_case = LLMTestCase(
        input          = input,
        actual_output  = actual_output,
        tools_called   = tools_called,
        expected_tools = expected_tools
    )

    test_cases.append(test_case)

tools_correctness = ToolCorrectnessMetric(strict_mode=False)
answer_revelancy  = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1-nano", include_reason=True)


# Run metrics
# Tools Accuracy
evaluate(
    test_cases     = test_cases, 
    metrics        = [tools_correctness, answer_revelancy],
    display_config = DisplayConfig(
                            verbose_mode=False,
                            show_indicator=True
                        )
)
print(f"TCS_LevensteinDist_avg : {sum_norm_lvs/count_cases}")