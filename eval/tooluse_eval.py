"Quick script for evaluating agent's tool use"
import asyncio
import json
import logging
import random

import tqdm
import dotenv
import textdistance
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.test_case import ToolCall
from deepeval.metrics   import ToolCorrectnessMetric
from deepeval.metrics   import AnswerRelevancyMetric
from typing import List

import numpy as np
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from together import Together

from langchain_core.messages import AIMessage
from langgraph.graph.graph import CompiledGraph

from agentic_sun_assistant.graph import create_and_compile_graph

dotenv.load_dotenv()


client      = Together()
eval_graph  = create_and_compile_graph()

# TODO: change this to args parsed later
# TODO: this is terrible, switch to functional programming
DATASET_PATH   = "eval/data/tool-calling/TC-R_Eval_cleaned.json"
EVAL_DATA_PATH = "eval_data.json"

with open(DATASET_PATH, 'r', encoding="utf-8") as eval_file:
    eval_dataset = json.load(eval_file)[:3]

async def ainfer_datapoint(input:str, eval_graph: CompiledGraph):
    """Run 1 eval datapoint and handle exceptions"""

    id_      = list(input.keys())[0]
    input_   = input[id_]
    aux_data = {"id": id_,"question": input_,}
    try:
        result = await eval_graph.ainvoke(
            {
                "messages":[{"type":"user", "content":input_}]
            }
        )
        return result, aux_data
    except Exception as e:
        print(f"Failed processing input {input_}, raised exception {e}")
        res = {"messages":[]}
        res.update(aux_data)
        return res, aux_data

async def ainfer_dataset(
        input_dataset: list[dict], eval_graph:CompiledGraph, 
        id_col_name:str = "QID",  init_question_col_name="Question"
    ) -> dict:
    """async run evaluation on all of the cases"""  

    # Extract user ins
    user_inputs  = [{_[id_col_name]:_[init_question_col_name]} for _ in input_dataset]
    # Run the inference
    run_results = await tqdm.asyncio.tqdm.gather(
        *[ainfer_datapoint(ui_, eval_graph) for ui_ in user_inputs]
    )

    return run_results

def flatten_tool_calls_trace(
    tools_seq :list[list[dict]]
):
    new_list = []
    for i, call_round in enumerate(tools_seq):
        tools_in_round = sorted([_["name"] for _ in call_round])
        new_list+=tools_in_round
    
    return new_list

def collect_tools_ins_tolist(tools_seq :list[list[dict]]):
    tool_info_parsing_map = {
        "Planner"      : (lambda x: str(x.get("full_text_plan", ""))),
        "RAG"          : (lambda x: str(x.get("query",[]))),
        "tavily_search": (lambda x: ','.join(x.get("queries", []))),
        "get_time_now" : None
    }
    res_dict = {} 
    default_parser = lambda x: RuntimeError("Unknown parser for this tool")
    for i_, call_round in enumerate(tools_seq):
        for tool_ in call_round:
            if tool_["name"] in tool_info_parsing_map.keys():

                tool_name       = tool_.get("name", "")
                tool_arg_parser = tool_info_parsing_map.get(tool_name, default_parser)
                if tool_arg_parser==None: 
                    continue

                tool_args_dict  = tool_.get("args", {})
                parsed_args     = tool_arg_parser(tool_args_dict)

                if tool_name not in res_dict.keys():
                    res_dict[tool_name] = [parsed_args]
                else:  
                    res_dict[tool_["name"]].append(parsed_args)

    return res_dict

def get_inputs_dict(dataset, q_):
    for dp_ in dataset:
        if dp_["Question"]==q_:
            return dp_
    return {}

async def inference_pipeline(
    dataset: list[dict],
    graph: CompiledGraph,
    inference_results_save_path: str
    ) -> None:
    """This is the main function for running eval pipeline, this results in a file being saved
    Inputs:
        - eval_graph: the graph with ainvoke() usable for inference
        - eval_data_save_path: absolute path
    """
    # Run inference
    inference_results = await ainfer_dataset(dataset, graph)
    res = []
    
    # Extract results, parse it into desired formatting
    for i_, output_ in enumerate(inference_results):
        new_result = {}
        id_data = output_[1]
        result  = output_[0]
        
        tool_sequence    = result.get("tool_calls_agg", [])
        reasoning_traces = result.get("reasoning_traces", ["No reasoning traces collected"])
        final_answer     = result.get("final_answer", "No answer collected.")
        
        # formatting to full texts
        reasoning_traces_str  = '\n'.join(reasoning_traces)
        tool_sequence_strlist = flatten_tool_calls_trace(tool_sequence)
        tools_input_agg_dict  = collect_tools_ins_tolist(tool_sequence)
        
        # Outputs setting
        output_agg = {} 
        output_agg["final_answer"]        = final_answer       
        output_agg["reasoning_texts_agg"] = reasoning_traces_str
        output_agg["tcs"]                 = tool_sequence_strlist
        output_agg["tools_called"]        = list(set(tool_sequence_strlist))
        output_agg["tools_inputs"]        = tools_input_agg_dict
        for tn_, inp_ in enumerate(output_agg["tools_inputs"]):
            if tn_=="RAG": output_agg["tools_inputs"]["RAG"]=','.join(inp_)
        
        new_result["output"] = output_agg
        
        # ID data
        input_dict = get_inputs_dict(dataset, id_data.get("question", ""))        
        id_agg     = {}
        id_agg["id"]                = input_dict.get("QID", random.randint(0, 100000))
        id_agg["question"]          = input_dict.get("Question")
        
        # Get ground-truths for merging to a single datapoint
        input_agg  = {}
        input_agg["reasoning_rubrics"] = input_dict.get("Reasoning", {}).get("Expected_Reasoning", "Failed getting reasoning.")
        input_agg["tcs"]               = input_dict.get("Tool_Call_Sequence_(TCS)", {}).get("Expected_TCS", ["Failed getting TCS"])
        input_agg["tools_names"]       = list(input_dict.get("Tool_Inputs").keys())

        input_agg["tools_inputs"] = {}
        for tool_name in input_agg["tools_names"]:
            expected_inputs = input_dict.get("Tool_Inputs", {}).get(tool_name, {}).get("Expected_Inputs", None)
            if expected_inputs == None:
                continue
            else: 
                input_agg["tools_inputs"][tool_name] = expected_inputs
        
        infered_datapoint = {"id_data": id_agg, "outputs":output_agg, "inputs": input_agg}
        res.append(infered_datapoint)
    
    with open(inference_results_save_path, 'w') as file:
        json.dump(res, file, indent=2)

asyncio.run(inference_pipeline(eval_dataset, eval_graph, "infered_data.json"))

#==================================================================================#
#===== This concludes the inference process, we must then assess this results =====#
#==================================================================================#


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