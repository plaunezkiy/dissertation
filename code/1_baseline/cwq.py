import pandas as pd
import csv
import os
from tqdm import tqdm
from utils.llm.mistral import MistralLLM
from utils.prompt import LEN_LIMITED_PROMPT_NO_CONTEXT
from utils.file import export_results_to_file


mistral = MistralLLM()

results = []
id_list = []
l = 0
experiment_name = f"bline"
res_path = f"/datasets/CWQ/results/{experiment_name}.csv"
if os.path.isfile(res_path):
    r_df = pd.read_csv(res_path)
    l = len(r_df)
    results = list(r_df.Model.values)

cwq = pd.read_csv("/datasets/CWQ/cwq-1000.csv", index_col=0)
for c, (i, r) in enumerate(tqdm(list(cwq.iterrows()))):
    id_list.append(i)
    if c < l:
        continue
    q = r.question
    prompt = LEN_LIMITED_PROMPT_NO_CONTEXT.format(question=q)
    response = mistral.get_response(prompt)
    results.append(response)

    if c % 10 == 0:
        export_results_to_file(res_path, results, id_list)
export_results_to_file(res_path, results, id_list)