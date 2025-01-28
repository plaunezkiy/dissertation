import csv
from rouge import Rouge
import pandas as pd
from tabulate import tabulate
import os
from utils.preprocessing import preprocess_text

rouge = Rouge()


class Dataset:
    def collect_knowledge_base(self):
        pass

    def generate_answers(self):
        pass
    
    def evaluate_answers(self):
        pass
    
    def tabulate_performance(self):
        pass


class FBQA_Dataset(Dataset):
    test_set_path = "/datasets/FreebaseQA/FreebaseQA-eval.json"
    result_set_paths = [
        "/datasets/FreebaseQA/results/bline.csv",
        "/datasets/FreebaseQA/results/bline2.csv",
        "/datasets/FreebaseQA/results/kb1.csv",
        "/datasets/FreebaseQA/results/kb2.csv",
        "/datasets/FreebaseQA/results/kb3.csv",
        "/datasets/FreebaseQA/results/sbert-kb1.csv",
        "/datasets/FreebaseQA/results/sbert-kb2.csv",
        "/datasets/FreebaseQA/results/sbert-kb3.csv",
    ]

    def __init__(self):
        self.test_set = pd.read_json(self.test_set_path)
        self.result_df = None

    @property    
    def answers(self):
        answers = self.test_set.Questions.apply(lambda t: t["Parses"][0]["Answers"][0]["AnswersName"][0])
        return answers

    @property
    def results(self):
        if self.result_df:
            return self.result_df
        results = [self.answers.tolist()]
        snames = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            snames.append(set_name)
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set)
            result_df.rename(columns={0: "Model"}, inplace=True)
            results.append(result_df.Model.tolist())
        result_df = pd.DataFrame(results, index=["Actual", *snames]).T
        result_df["quid"] = self.test_set.Questions.apply(lambda t: t.get("Question-ID"))
        self.result_df = result_df
        return self.result_df
    
    def check_correct(self, model_answer, actual_answer):
        m = preprocess_text(model_answer)
        a = preprocess_text(actual_answer)
        if m and a:
            scores = rouge.get_scores(m[:300], a)[0]
            return scores["rouge-l"]["r"] >= 0.5
        return False

    def evaluate_answers(self):
        results = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set)
            result_df.rename(columns={0: "Model"}, inplace=True)
            result_df["Actual"] = self.answers
            result_df["Model"] = result_df["Model"].fillna(" ").apply(lambda s: s.lower())
            # .apply(lambda t: t.iloc[1], axis=1))
            result_df["Correct"] = result_df.apply(lambda t: t.iloc[1] in t.iloc[0], axis=1)
            accuracy = sum(result_df.Correct) / len(result_df)
            # add to the list
            results.append([set_name, accuracy])
        return results
    
    def tabulate_performance(self):
        results = self.evaluate_answers()
        print("FBQA")
        print(tabulate(results, tablefmt="grid", headers=["Method", "Test Set"]))
            

class MetaQA_Dataset(Dataset):
    test_set_path = "/datasets/MetaQA/{hop}/qa_test.txt"
    hops = ["1hop", "2hop", "3hop"]
    result_set_paths = [
        "/datasets/MetaQA/results/{hop}/bline.csv",
        # "/datasets/MetaQA/results/{hop}/bline2.csv",
        "/datasets/MetaQA/results/{hop}/kb1.csv",
        "/datasets/MetaQA/results/{hop}/kb2.csv",
        "/datasets/MetaQA/results/{hop}/kb3.csv",
        "/datasets/MetaQA/results/{hop}/sbert-kb3.csv",
    ]

    def __init__(self):
        test_sets = {}
        for hop in self.hops:
            tset_path = self.test_set_path.format(hop=hop)
            tset = pd.read_csv(tset_path, sep="\t", header=None)
            tset.rename(columns={0: "Question", 1: "Answers"}, inplace=True)
            tset.Answers = tset.apply(lambda t: set(t.Answers.lower().split("|")), axis=1)
            test_sets[hop] = tset
        self.test_sets = test_sets
    
    def get_rouge_score_for_answers(self, actual_answers, model_answers):
        """
        Calculate ROUGE score between every actual and every model answer.
        Pick the highest one and return as final.
        """
        max_r = 0
        for a in actual_answers:
            for m in model_answers:
                if m and a:
                    scores = rouge.get_scores(m[:300], a)[0]
                    if scores["rouge-l"]["r"] > max_r:
                        max_r = scores["rouge-l"]["r"]
        return max_r
    
    def answers(self, hop):
        return self.test_sets[hop].Answers

    def evaluate_answers(self):
        results = {}
        for hop in self.hops:
            results[hop] = []
            for r_set in self.result_set_paths:
                set_name = r_set.split("/")[-1].split(".")[0]
                # if the file does not exist, 0 accuracy
                if not os.path.exists(r_set.format(hop=hop)):
                    results[hop].append([set_name, 0])
                    continue
                # otherwise, compute the accuracy
                # merge copy of test set with results
                result_df = self.test_sets[hop].copy()
                r_df = pd.read_csv(r_set.format(hop=hop))
                result_df.insert(2, "Model", r_df.Model)
                # account for empty entries
                result_df.fillna("", inplace=True)
                result_df["Model"] = result_df.apply(lambda t: set(t.Model.lower().split("|")), axis=1)
                # compute rouge-match and correctness
                # print(result_df)
                result_df["rouge-l"] = result_df.apply(lambda t: self.get_rouge_score_for_answers(t.Answers, t.Model), axis=1)
                result_df["Correct"] = result_df.apply(lambda t: t["rouge-l"] >= 0.5, axis=1)
                accuracy = sum(result_df.Correct) / len(result_df)
                # add to the list
                results[hop].append([set_name, accuracy])
        return results
    
    def tabulate_performance(self):
        """
        [
            ["Method", "1hop", "2hop", "3hop"]
            ["bline", 0, 0, 0]
            ["kb1", 0, 0, 0]
        ]
        """
        header = ["Method"]
        results = self.evaluate_answers()
        table = []
        for method_path in self.result_set_paths:
            method = method_path.split("/")[-1].split(".")[0]
            table.append([method])
        
        for hop in self.hops:
            header.append(hop)
            # method, accuracy
            for i, (method, accuracy) in enumerate(results[hop]):
                table[i].append(accuracy)

        print("MetaQA")
        print(tabulate(table, tablefmt="grid", headers=header))
