import asyncio
import json
import logging

import tqdm
import dotenv
import textdistance
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.test_case import ToolCall
from deepeval.metrics   import ToolCorrectnessMetric
from deepeval.metrics   import AnswerRelevancyMetric
from deepeval.evaluate  import DisplayConfig
from typing import List

import numpy as np
import pandas as pd

from langchain_core.messages import AIMessage
from langgraph.graph.graph import CompiledGraph

from agentic_sun_assistant.graph import create_and_compile_graph

# Set up envs
dotenv.load_dotenv()
eval_graph   = create_and_compile_graph()

# TODO: Add args parsing
# TODO: change this to args parsed later
DATASET_PATH = "eval/data/tool-calling/TC-R_Eval_cleaned.json"
eval_dataset = json.load(open(DATASET_PATH, 'r'))
# TODO: this is terrible, switch to functional programming



# STEP 1  : Run Inference for Inputs [DONE]
# STEP 2  : Collect Infered Data, Merge with Input [DONE]

# STEP 3  : Run Evaluation on data
# STEP 3.1: Tool Accuracy [Doing]
# STEP 3.2: TCS Levenstein [Done]
# STEP 3.3: Tool Input Correctness [Done]
# STEP 3.4: Reasoning Trace Eval (TBD)

# STEP 4  : Export Report


# STEP 1: Run Inference for Inputs
async def arun_datapoint(input:str, eval_graph: CompiledGraph):
    """Run 1 eval datapoint and handle exceptions"""

    try:
        result = await eval_graph.ainvoke(
            {
                "messages":[{"type":"user", "content":input}]
            }
        )
        print(result["tool_calls_agg"])
        return result
    except Exception as e:
        print(f"Failed processing input {input}, raised exception {e}")
        return {
            "messages":[]
        }


async def arun_dataset(
        eval_dataset: dict, eval_graph:CompiledGraph, eval_data_path: str = 'eval_data.json'
    ) -> dict:
    """async run evaluation on all of the cases"""

    user_inputs  = [_["Question"] for _ in eval_dataset]
    eval_results = [{} for _ in user_inputs]
    
    run_results = await tqdm.asyncio.tqdm.gather(
        *[arun_datapoint(ui_, eval_graph) for ui_ in user_inputs]
    )
    
    for i in range(len(user_inputs)):
        print(run_results[i]["tool_calls_agg"])
        eval_results[i]["id"] = eval_dataset[i]["QID"]
        eval_results[i]["user_input"]   = user_inputs[i]
        
        ideal_tools  = eval_dataset[i]["Tools_Accuracy"]["Expected_Tools"]
        if ideal_tools == None:
            ideal_tools = []
        eval_results[i]["expected_tools"] = {}
        for tool_name in ideal_tools:
            if 'get_time_now' in tool_name: continue
            eval_results[i]["expected_tools"][tool_name] = eval_dataset[i]["Tool_Inputs"][tool_name]["Expected_Inputs"]
        
        if eval_dataset[i]["Tool_Call_Sequence_(TCS)"]["Expected_TCS"] == None:
            eval_dataset[i]["Tool_Call_Sequence_(TCS)"]["Expected_TCS"] = []
        eval_results[i]["expected_tcs"] = eval_dataset[i]["Tool_Call_Sequence_(TCS)"]["Expected_TCS"]
        eval_results[i]["actual_tcs"]   = run_results[i]["tool_calls_agg"]

        # eval_results[i]["actual_tcs"]   = getattr(run_results[i], "tool_calls_agg", [])
    
    with open(eval_data_path, 'w') as file:
        json.dump(eval_results, file, indent=2)
    
    logging.info

# EVAL_DATA_PATH = "eval/data/tool-calling/TC-R_Eval_cleaned_merged.json"
EVAL_DATA_PATH = "eval_data.json"
asyncio.run(arun_dataset(eval_dataset, eval_graph, EVAL_DATA_PATH))

# STEP 3  : Run Evaluation on data
test_cases   = []
sum_norm_lvs = 0
count_cases  = 0

def _base_levenshtein(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0  # Same base label
            else:
                cost = 1  # Different base labels
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[n][m]

def _calc_levenstein(expected_tcs: List[str], actual_tcs:List[List[dict]]):
    flattened_tcs = [
        item["name"]
        for sublist in actual_tcs
        for item in sorted(sublist, key=lambda x: x["name"])
    ]
    return _base_levenshtein(expected_tcs, flattened_tcs)

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Set your OpenAI API key
from together import Together

client = Together()

def get_embedding(text: str, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model="BAAI/bge-large-en-v1.5",
        input=text
        )

    return response.data[0].embedding

def recall_by_embedding_similarity(list_predict, list_actual, threshold=0.85):
    if not list_actual:
        return 0.0  # Avoid division by zero

    # Get embeddings
    embeddings_pred = [get_embedding(pred) for pred in list_predict]
    embeddings_actual = [get_embedding(act) for act in list_actual]

    # Calculate true positives
    true_positive = 0
    for actual_emb in embeddings_actual:
        max_sim = max(
            cosine_similarity([actual_emb], [pred_emb])[0][0]
            for pred_emb in embeddings_pred
        )
        if max_sim > threshold:
            true_positive += 1

    recall = true_positive / len(list_actual)
    return recall


def _eval_datapoint(datapoint):

    dp_id = datapoint["id"]
    dp_ui = datapoint["user_input"]
    
    tcs_levenstein_dist = _calc_levenstein(datapoint["expected_tcs"], datapoint["actual_tcs"])
    # if "RAG" in datapoint["expected_tools"].keys():
        
    #     RAG_inputs_agg = 
    rag_ins  = ""
    ws_ins   = ""
    plnn_ins = ""
    
    for inter_ in datapoint["actual_tcs"]:
        for call_ in inter_:
            if call_["name"] == "RAG": rag_ins+=call_["args"]["query"]
            if call_["name"] == "tavily_search": ws_ins+=','.join(call_["args"]["queries"])
            if call_["name"] == "Planner": plnn_ins+=call_["args"]["full_plan_text"]
    
    tool_ins_res = {}

    tool_names = ["RAG", "tavily_search", "Planner"]
        
    if "RAG" in datapoint["expected_tools"].keys():
        if len(rag_ins)==0:
            tool_ins_res["input_RAG_sim"]=1
        else: tool_ins_res["input_RAG_sim"] = recall_by_embedding_similarity([rag_ins], [datapoint["expected_tools"]["RAG"]])
    if "tavily_search" in datapoint["expected_tools"]:
        if len(ws_ins)==0:
            tool_ins_res["input_WS_sim"]=1
        else: tool_ins_res["input_WS_sim"] = recall_by_embedding_similarity([ws_ins], [datapoint["expected_tools"]["tavily_search"]])
    if "Planner" in datapoint["expected_tools"]:
        if len(plnn_ins)==0:
            tool_ins_res["input_Pln_sim"]=1
        else: tool_ins_res["input_Pln_sim"] = recall_by_embedding_similarity([plnn_ins], [datapoint["expected_tools"]["Planner"]])

    print(tool_ins_res)

    return {
        "QID":dp_id,
        "Question": dp_ui,
        "TCS_Levenstein": tcs_levenstein_dist,
        "tool_inputs": tool_ins_res
    }
    
eval_results          = json.load(open(EVAL_DATA_PATH, 'r'))
levenstein_dist_total = 0

def _eval_all(
    eval_results: List[dict]
):
    res = []
    for datapoint in eval_results:
        res.append(_eval_datapoint(datapoint))
    
    print(res)
    return pd.DataFrame(res)


_eval_all(eval_results).to_csv("per_row_report.csv")

# print(f"Avg Levenstein: {levenstein_dist_total/len(eval_results)}")

# for run_ in eval_results:

#     tools_called = []
#     for msg in run_["state"]["messages"]:
#         tools_called += getattr(msg, "tool_calls", [])

#     tools_called = [_["name"] for _ in tools_called]
#     print(tools_called)

#     # Collect data for evaluation
#     input          = run_["user_input"]
#     actual_output  = run_["state"]["messages"][-1].content if len(run_["state"]["messages"])>0 else ""
#     tools_called   = [ToolCall(name=n) for n in list(set(tools_called))]
#     expected_tools = [ToolCall(name=n) for n in run_["ideal_tools"]] if run_["ideal_tools"] is not None else []

#     # Skip cases that does not require a tool chain
#     expected_tcs  = run_["tcs"] if run_["tcs"] is not None else []
#     if expected_tcs is not []:
#         tcs           = tools_called
#         max_len       = max(len(expected_tcs), len(tcs))
#         lvs_dist      = textdistance.levenshtein.distance(tcs,expected_tcs)
#         if max_len == 0:
#             norm_lvs_dist = 1    
#         else: norm_lvs_dist = lvs_dist/max_len
#         sum_norm_lvs += norm_lvs_dist
#         count_cases  += 1

#     test_case = LLMTestCase(
#         input          = input,
#         actual_output  = actual_output,
#         tools_called   = tools_called,
#         expected_tools = expected_tools
#     )

#     test_cases.append(test_case)

# tools_correctness = ToolCorrectnessMetric(strict_mode=False)
# answer_revelancy  = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1-nano", include_reason=True)


# # Run metrics
# # Tools Accuracy
# evaluate(
#     test_cases     = test_cases, 
#     metrics        = [tools_correctness, answer_revelancy],
#     display_config = DisplayConfig(
#                             verbose_mode=False,
#                             show_indicator=True
#                         )
# )
# print(f"TCS_LevensteinDist_avg : {sum_norm_lvs/count_cases}")