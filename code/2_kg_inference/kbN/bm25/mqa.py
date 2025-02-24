import gc
import re
import csv
import os
import gc
import torch
# device = torch.device("cuda:1")
# torch.cuda.set_device(device)
# torch.set_default_device(device)
import Stemmer
import pandas as pd
from tqdm import tqdm
from utils.graph import KGraphPreproc
from utils.graph.chain import GraphChain
from utils.llm.mistral import MistralLLM
from utils.prompt import GRAPH_QA_PROMPT, ENTITY_PROMPT
from utils.file import export_results_to_file

# set to "cuda:1" for running in parallel on both GPUs

def get_response(prompt):
    # global chain
    # del mistral
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]


metaqa_graph = KGraphPreproc.get_metaqa_graph()
mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral,
    graph=metaqa_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)
chain.ranking_strategy = "sbert"

####### load the qa, graph, llm
for hop in ["1hop", "2hop", "3hop"]:
    print(hop)
    # load q's
    metaqa = pd.read_csv(f"/datasets/MetaQA/{hop}/test_1000.txt", header=None, index_col=0)
    metaqa.rename(columns={1: "Question", 2: "Answers"}, inplace=True)


    for depth in [1, 3]:
        print(f"depth: {depth}")
        # set the depth
        chain.exploration_depth = depth
        # init experiment
        experiment_name = f"kb{depth}"
        res_path = f"/datasets/MetaQA/results/{hop}/{experiment_name}.csv"
        results = []
        id_list = []
        l = 0
        # load if preinit'ed
        if os.path.isfile(res_path):
            r_df = pd.read_csv(res_path)
            l = len(r_df)
            results = list(r_df.Model.values)
        # run through
        for c, (i, r) in enumerate(tqdm(list(metaqa.iterrows()))):
            id_list.append(i)
            if c < l:
                continue
            q = r.Question
            response = get_response(q)
            results.append(response)
            # backup every 10 qs
            if c % 10 == 0:
                export_results_to_file(res_path, results, id_list)
        export_results_to_file(res_path, results, id_list)