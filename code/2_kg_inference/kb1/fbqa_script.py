import pandas as pd
import torch
import gc
from tqdm import tqdm
import csv
import gc
from utils.graph import KGraphPreproc
from utils.file import export_results_to_file

fbqa = pd.read_json("/datasets/FreebaseQA/FreebaseQA-eval.json")
def get_fbqa_data(question_row):
    """
    Takes in a dataset row and returns Q and A as strings
    """
    question = question_row.Questions.get("RawQuestion", None)
    parse = question_row.Questions.get("Parses", [None])[0]
    if not parse:
        print(f"error in question: {question}")
        return question, None
    answer = parse.get("Answers")
    return question, answer

####### loading kb
fbkb_graph = KGraphPreproc.get_fbkb_graph()

###### inference
from langchain_core.prompts.prompt import PromptTemplate

EXTRACT_ENTITIES_TEMPLATE = """
Below is a question, extract all entities and return them separeted by a coma.
Only return the entities with no extra text and separators. 
Do not introduce or deduce anything not present in the question.
Question:
{question}"""

entity_prompt = PromptTemplate(
    input_variables=["question"],
    template=EXTRACT_ENTITIES_TEMPLATE
)

LEN_LIMITED_PROMPT = """
You will be given a context containing relevant entity triplets and a question.

Using the context, you should answer the question below in under a sentence with no other text but the answer.
Context:
{context}

Question:
{question}
"""

GRAPH_QA_PROMPT = PromptTemplate(
    template=LEN_LIMITED_PROMPT, input_variables=["context", "question"]
)

mistral = MistralLLM()
chain = GraphChain.from_llm(
    llm=mistral, 
    graph=fbkb_graph, 
    qa_prompt=GRAPH_QA_PROMPT,
    entity_prompt=entity_prompt,
    verbose=False,
)


def get_response(prompt):
    global chain
    # del mistral
    gc.collect()
    torch.cuda.empty_cache()
    r = chain.invoke(prompt)
    return r["result"]


###### tests


bline_path = "/datasets/FreebaseQA/results/kb1.csv"
bline = pd.read_csv(bline_path)
l = len(bline)
baseline_results = list(bline.Model.values)

fbqa = pd.read_json("/datasets/FreebaseQA/FreebaseQA-eval.json")
# fbqa.Questions[0].get("RawQuestions", None)
results = []
for i, r in tqdm(list(fbqa.iterrows())):
    if i < l:
        continue
    q, a = get_fbqa_data(r)
    # 
    # print("Question:", q)
    # print("Answer:", a)
    # print("Prompt:", prompt)
    response = get_response(q)
    # print("Model:", response)
    results.append(response)

    if i % 250 == 0:
        export_results_to_file(bline_path, results)
export_results_to_file(bline_path, results)
