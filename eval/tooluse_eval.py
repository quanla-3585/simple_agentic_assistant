"Quick script for evaluating agent's tool use"
import asyncio
import json
import logging
import random

import tqdm.asyncio # for some reason this is where tqdm async mode is, honestly dont know why

import dotenv
from typing import List

import pandas as pd
from together import Together

from langgraph.graph.graph import CompiledGraph

from agentic_sun_assistant.graph import create_and_compile_graph
from utils import calc_tools_acc
from utils import calc_tcs_levenstein
from utils import recall_by_embedding_similarity

dotenv.load_dotenv()

#==================================================================================#
#===== Get variables and set global variables =====================================#
#==================================================================================#

client      = Together()
eval_graph  = create_and_compile_graph()
tool_info_parsing_map = {
    "Planner"      : (lambda x: str(x.get("full_text_plan", ""))),
    "RAG"          : (lambda x: str(x.get("query",[]))),
    "tavily_search": (lambda x: ','.join(x.get("queries", []))),
    "get_time_now" : None
}

# TODO: change this to args parsed later
# TODO: this is terrible, switch to functional programming
DATASET_PATH      = "eval/data/tool-calling/TC-R_Eval_cleaned.json"
INFERED_DATA_PATH = "infered_data.json"
REPORT_FILE_PATH  = "per_row_report.csv"

with open(DATASET_PATH, 'r', encoding="utf-8") as eval_file:
    eval_dataset = json.load(eval_file)

#==================================================================================#
#===== Start Inference Process ====================================================#
#==================================================================================#

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
            if tn_=="RAG":
                logging.warning(f"Found multiple RAG calls with concat input {tn_, inp_}")
                output_agg["tools_inputs"]["RAG"]=','.join(inp_)
        
        new_result["output"] = output_agg
        
        # ID data
        input_dict = get_inputs_dict(dataset, id_data.get("question", ""))        
        id_agg     = {}
        id_agg["id"]                = input_dict.get("QID", random.randint(0, 100000))
        id_agg["question"]          = input_dict.get("Question")
        
        # Get ground-truths for merging to a single datapoint
        input_agg  = {}
        input_agg["reasoning_rubrics"] = input_dict.get("Reasoning", {}).get("Expected_Reasoning", "Failed getting reasoning.")
        input_agg["tcs"]               = input_dict.get("Tool_Call_Sequence_(TCS)", {}).get("Expected_TCS", [])
        if input_agg["tcs"] == None: input_agg["tcs"] = []
        input_agg["tools_names"]       = list(set(input_agg["tcs"]))

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
        json.dump(res, file, indent=2, ensure_ascii=False)

asyncio.run(inference_pipeline(eval_dataset, eval_graph, "infered_data.json"))

#==================================================================================#
#===== Start Evaluation Process ===================================================#
#==================================================================================#

test_cases   = []
sum_norm_lvs = 0
count_cases  = 0

def _eval_datapoint(datapoint):

    dp_id = datapoint["id_data"]["id"]
    dp_ui = datapoint["id_data"]["question"]

    tools_called   = datapoint["outputs"].get("tools_called", [])
    tools_expected = datapoint["inputs"].get("tools_names", []) 

    tcs_called   = datapoint["outputs"].get("tcs", [])
    tcs_expected = datapoint["inputs"].get("tcs", [])

    args_expected = datapoint["inputs"].get("tools_inputs", {})
    args_llm      = datapoint["outputs"].get("tools_inputs", {})
    
    # tcs calcs
    tool_acc            = calc_tools_acc(tools_called, tools_expected)
    tcs_levenstein_dist = calc_tcs_levenstein(tcs_expected, tcs_called)
    
    # Inputs calcs
    tp_tools = list(set(tools_called).intersection(set(tools_expected)))
    print(f"TP TOOLS: {tools_called, tools_expected}")
    inputs_eval_dict = {}
    for tn_ in tp_tools:
        if "get_time_now" in tn_: continue
        if tn_ == "RAG" or tn_ == "tavily_search":
            args_llm_ = args_llm.get(tn_, [])
        else:
            args_llm_ = [args_llm.get(tn_, "None")]
        args_expected_ = [args_expected.get(tn_, "None")]
        try:
            inputs_eval_dict[f"{tn_}_args_recall"] = recall_by_embedding_similarity(args_llm_, args_expected_, 0.5)
        except:
            inputs_eval_dict[f"{tn_}_args_recall"] = 0
    res = {
        "question": dp_ui,
        "tcs_actual": tcs_called,
        "tcs_expected": tcs_expected,
        "tools_called_acc": tool_acc,
        "tcs_levenstein_dist": tcs_levenstein_dist
    }

    res.update(inputs_eval_dict)
    print(f"INPUT EVAL: {inputs_eval_dict}")
    
    return res

def eval_all(
    eval_results: List[dict]
):
    res = []
    for datapoint in eval_results:
        res_dp = _eval_datapoint(datapoint)
        res.append(res_dp)
    
    print(res)
    return pd.DataFrame(res)

infered_dataset   = json.load(open(INFERED_DATA_PATH, 'r'))
evaluation_result = eval_all(infered_dataset)

#==================================================================================#
#===== Save to report, we are currently using a CSV ===============================#
#==================================================================================#

evaluation_result.to_csv(REPORT_FILE_PATH)