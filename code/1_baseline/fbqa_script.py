import pandas as pd
import csv
from tqdm import tqdm
from utils.llm.mistral import MistralLLM
from utils.prompt import LEN_LIMITED_PROMPT_NO_CONTEXT
from utils.file import export_results_to_file


mistral = MistralLLM()

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


bline_path = "/datasets/FreebaseQA/results/bline.csv"
bline = pd.read_csv(bline_path)
l = len(bline)
baseline_results = list(bline.Model.values)

fbqa = pd.read_json("/datasets/FreebaseQA/FreebaseQA-eval.json")
# fbqa.Questions[0].get("RawQuestions", None)
baseline_results = []
for i, r in tqdm(list(fbqa.iterrows())):
    if i < l:
        continue
    q, a = get_fbqa_data(r)
    prompt = LEN_LIMITED_PROMPT_NO_CONTEXT.format(question=q)
    response = mistral.get_response(prompt)
    baseline_results.append(response)

    if i % 250 == 0:
        export_results_to_file
export_results_to_file

