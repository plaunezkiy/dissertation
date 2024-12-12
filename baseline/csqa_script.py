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



csqa = pd.read_csv("/datasets/CosmosQA/valid.csv")


CSQA_PROMPT = """
Below is the relevant context, a question and 4 candidate answers.
Given the context, return the letter of the most plausible answer.
Only return the letter of the answer and nothing else.
CONTEXT:
{context}
QUESTION:
{question}
CANDIDATE ANSWERS:
A) {A}
B) {B}
C) {C}
D) {D}
"""

bline = pd.read_csv("/datasets/CosmosQA/results/bline.csv")
l = len(bline)
baseline_results = list(bline.Model.values)

for i, r in tqdm(list(csqa.iterrows())):
    if i < l:
        continue
    prompt = CSQA_PROMPT.format(
        context=r.context,
        question=r.question,
        A=r.answer0,
        B=r.answer1,
        C=r.answer2,
        D=r.answer3
    )
    # print("Prompt:", prompt)
    response = get_response(prompt)
    # print("Model:", response)
    baseline_results.append(response)
    
    if i % 250 == 0:
        with open(f"/datasets/CosmosQA/results/bline.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"])
            for r in baseline_results:
                writer.writerow([str(r)])

with open(f"/datasets/CosmosQA/results/bline.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Model"])
    for r in baseline_results:
        writer.writerow([str(r)])