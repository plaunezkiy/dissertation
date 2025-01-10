import pandas as pd
import csv
import os
from tqdm import tqdm
import torch
import gc
from langchain_community.graphs.networkx_graph import KnowledgeTriple
from langchain.chains import GraphQAChain
from utils.llm.mistral import MistralLLM
from utils.prompt import ENTITY_PROMPT, GRAPH_QA_PROMPT
from utils.graph import KGraphPreproc
from utils.file import export_results_to_file


metaqa_kb = pd.read_csv("/datasets/MetaQA/KB/kb.txt", sep="|", header=None)
metaqa_kb.rename(columns={0: "subject", 1: "relation", 2: "object"}, inplace=True)
# construct the KG
mqa_graph = KGraphPreproc()
for i, r in metaqa_kb.iterrows():
    triplet = KnowledgeTriple(
        r.subject,
        r.relation,
        r.object,
    )
    mqa_graph.add_triple(triplet)
# prep the graph
mqa_graph.generate_preprocessed_nodes()
# account for directional edges (llm will figure it out)
mqa_graph._graph = mqa_graph._graph.to_undirected()

# set up inference
mistral = MistralLLM()
chain = GraphQAChain.from_llm(
    llm=mistral, 
    graph=mqa_graph,
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=ENTITY_PROMPT,
    verbose=False,
)

def get_response(prompt):
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]

for hop in ["1hop", "2hop", "3hop"]:
    print(hop)
    # load the qset
    metaqa = pd.read_csv(f"/datasets/MetaQA/{hop}/qa_test.txt", sep="\t", header=None)
    metaqa.rename(columns={0: "Question", 1: "Answers"}, inplace=True)
    # init for results
    results = []
    l = 0
    results_path = f"/datasets/MetaQA/results/{hop}/kb1.csv"
    # check if some results are already cached
    if os.path.exists(results_path):
        res_df = pd.read_csv(results_path)
        l = len(res_df)
        results = list(res_df.Model.values)
    # go through qset
    for i, r in tqdm(list(metaqa.iterrows())):
        if i < l:
            continue
        q = r.Question
        response = get_response(q)
        results.append(response)
        # cache the result
        if i % 250 == 0:
            export_results_to_file(results_path, results)
    export_results_to_file(results_path, results)