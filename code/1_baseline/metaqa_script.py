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

    out_tokens, _ = generate([tokens], model, max_tokens=4096, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result


for hop in ["1hop", "2hop", "3hop"]:
    print(hop)
    metaqa = pd.read_csv(f"/datasets/MetaQA/{hop}/qa_test.txt", sep="\t", header=None)
    metaqa.rename(columns={0: "Question", 1: "Answers"}, inplace=True)
    metaqa.Answers = metaqa.apply(lambda t: t.Answers.split("|"), axis=1)
    
    bline_path = f"/datasets/MetaQA/results/{hop}/bline2.csv"
    bline = pd.read_csv(bline_path)
    l = len(bline)
    baseline_results = list(bline.Model.values)


    LEN_LIMITED_PROMPT = """
    You should answer the question below in under a sentence with no other text but the answer.
    If there are multiple answers, separate the answers by "|".
    Do not include "Answer:". Produce the answer only.
    Question:
    {question}
    """

    

    for i, r in tqdm(list(metaqa.iterrows())):
        if i < l:
            continue
        q = r.Question
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