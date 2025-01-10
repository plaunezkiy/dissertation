import pandas as pd
import csv
from tqdm import tqdm
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


tokenizer = MistralTokenizer.from_file("/models/M7B/tokenizer.model.v3")  # change to extracted tokenizer file
model = Transformer.from_folder("/models/M7B")  # change to extracted model dir



def get_response(prompt):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result


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


LEN_LIMITED_PROMPT = """
You should answer the question below in under a sentence with no other text but the answer.
Question:
{question}
"""

bline_path = "/datasets/FreebaseQA/results/bline2.csv"
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
    # 
    # print("Question:", q)
    # print("Answer:", a)
    prompt = LEN_LIMITED_PROMPT.format(question=q)
    # print("Prompt:", prompt)
    response = get_response(prompt)
    # print("Model:", response)
    baseline_results.append(response)

    if i % 250 == 0:
        with open(bline_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"])
            for r in baseline_results:
                writer.writerow([str(r)])

with open(bline_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Model"])
    for r in baseline_results:
        writer.writerow([str(r)])

