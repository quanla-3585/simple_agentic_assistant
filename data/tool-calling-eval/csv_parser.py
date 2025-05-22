import pandas as pd

DATASET_CSV_FILEPATH = "data/tool-calling-eval/ToolCall-ReasoningEvaluationSet.csv"

# Step 1: Load just the header rows (first 3 rows) as strings and sanitize 'Unnamed' columns
raw_header = pd.read_csv(DATASET_CSV_FILEPATH, nrows=3, header=None, dtype=str)
raw_header = raw_header.replace(r"^Unnamed.*", pd.NA, regex=True).ffill(axis=1)

# Step 2: Create list of tuples representing columns
column_tuples = list(zip(*raw_header.values))

# Step 3: Build clean dot-separated column names with apostrophes removed
flat_columns = []
for col in column_tuples:
    parts = [str(lvl).strip() if not pd.isna(lvl) else '' for lvl in col]

    # Drop spurious 'Recall' level for Reasoning columns
    if parts[0] == 'Reasoning' and parts[-1] == 'Recall':
        parts = parts[:-1]

    # Join and clean
    col_name = '.'.join(filter(None, parts))
    col_name = col_name.replace(' ', '_').replace("'", "")
    flat_columns.append(col_name)

# Step 4: Load the full dataset *without* header rows, since we already processed them
raw_dataset = pd.read_csv(DATASET_CSV_FILEPATH, header=3)  # skip first 3 rows (headers)

# Step 5: Assign cleaned column names to dataset
raw_dataset.columns = flat_columns

print(raw_dataset.columns)

# print(raw_dataset["Question"])
raw_dataset.to_csv("./data/tool-calling-eval/TC-R_Eval_cleaned.csv")
question_col_name = "Question"

import json

def nested_dict_from_flat(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split('.')
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested

import math
def preprocess_row(row):
    d = {}
    for k, v in row.items():
        # Convert NaN floats to None
        if isinstance(v, float) and math.isnan(v):
            d[k] = None
        # Split comma-separated strings into lists for target keys
        elif k in ["Tools_Accuracy.Expected_Tools", "Tool_Call_Sequence_(TCS).Expected_TCS"] and isinstance(v, str):
            d[k] = [x.strip() for x in v.split(",")]
        else:
            d[k] = v
    return d

json_list = [nested_dict_from_flat(preprocess_row(row)) for _, row in raw_dataset.iterrows()]

# Export to JSON file
with open("./data/tool-calling-eval/TC-R_Eval_cleaned.json", "w") as f:
    json.dump(json_list, f, indent=2)
