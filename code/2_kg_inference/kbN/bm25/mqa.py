import gc
import re
import csv
import os
import gc
import torch
device = torch.device("cuda:1")
torch.cuda.set_device(device)
torch.set_default_device(device)
import Stemmer
import pandas as pd
from tqdm import tqdm
from utils.graph import KGraphPreproc
from utils.graph.chain import GraphChain
from utils.llm.mistral import MistralLLM
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT

# set to "cuda:1" for running in parallel on both GPUs

def get_response(prompt):
    # global chain
    # del mistral
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]


def save_results(fpath, data_rows):
    with open(fpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"])
        for r in data_rows:
            writer.writerow([str(r)])

metaqa_graph = KGraphPreproc.get_metaqa_graph()
mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral,
    graph=metaqa_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)

####### load the qa, graph, llm
for hop in ["3hop"]:
    print(hop)
    # load q's
    metaqa = pd.read_csv(f"/datasets/MetaQA/{hop}/qa_test.txt", sep="\t", header=None)
    metaqa.rename(columns={0: "Question", 1: "Answers"}, inplace=True)


    for depth in [2,3]:
        print(f"depth: {depth}")
        # set the depth
        chain.exploration_depth = depth
        # init experiment
        experiment_name = f"kb{depth}"
        res_path = f"/datasets/MetaQA/results/{hop}/{experiment_name}.csv"
        results = []
        l = 0
        # load if preinit'ed
        if os.path.isfile(res_path):
            r_df = pd.read_csv(res_path)
            l = len(r_df)
            results = list(r_df.Model.values)
        # run through
        for i, r in tqdm(list(metaqa.iterrows())):
            if i < l:
                continue
            q = r.Question
            response = get_response(q)
            results.append(response)
            # backup every 10 qs
            if i % 10 == 0:
                save_results(res_path, results)
        save_results(res_path, results)