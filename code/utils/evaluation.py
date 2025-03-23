import csv


import pandas as pd
from tabulate import tabulate
import ast
import os
from utils.preprocessing import preprocess_text
from utils.scores import calculate_em_accuracy, calculate_f1_accuracy
import numpy as np
import re



class Dataset:
    def collect_knowledge_base(self):
        pass

    def generate_answers(self):
        pass
    
    def evaluate_answers(self):
        pass
    
    def tabulate_performance(self):
        pass


class CWQ_Dataset(Dataset):
    test_set_path = "/datasets/CWQ/cwq-1000.csv"
    result_set_paths = [
        "/datasets/CWQ/results/bline.csv",
        "/datasets/CWQ/results/bline2.csv",
        "/datasets/CWQ/results/kb-path.csv",
        *[
            f"/datasets/CWQ/results/kb{d}.csv" 
            for d in range(1, 8)
        ],
        *[
            f"/datasets/CWQ/results/sbert-kb{d}.csv" 
            for d in range(1, 8)
        ],
        "/datasets/CWQ/results/tog-lp-4.csv",
    ]

    def __init__(self):
        self.test_set = pd.read_csv(self.test_set_path, index_col=0)
        self.test_set["topic_ids"] = self.test_set["topic_ids"].apply(ast.literal_eval)
        self.test_set["answer_ids"] = self.test_set["answer_ids"].apply(ast.literal_eval)
        self.result_df = None

    @property
    def answers(self):
        def extract_answers(answer_list):
            a_list = ast.literal_eval(answer_list)
            return list(map(lambda x: preprocess_text(x["answer"]), a_list))
        return self.test_set.answers.apply(extract_answers).values

    @property
    def results(self):
        if self.result_df is not None and not self.result_df.empty:
            return self.result_df
        results = [self.answers.tolist()]
        snames = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            snames.append(set_name)
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set)
            # 
            result_df.rename(columns={0: "Model"}, inplace=True)
            results.append(result_df.Model.tolist())
        result_df = pd.DataFrame(results, index=["Actual", *snames]).T
        self.result_df = result_df
        return self.result_df

    def evaluate_answers(self):
        results = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set, dtype=str)
            result_df.rename(columns={0: "Model"}, inplace=True)
            values = result_df.Model.apply(lambda s: str(s).lower().split(",")).values
            # values = np.pad(values, (0, 1000 - values.size), constant_values="")
            # result_df.reset_index(inplace=True)
            result_df["Model"] = values
            result_df = result_df.reindex(range(1000), fill_value="")
            # print(set_name, len(result_df))
            result_df["Actual"] = self.answers
            # 
            result_df["EM"] = result_df.apply(lambda t: calculate_em_accuracy(t.Actual, t.Model), axis=1)
            result_df["F1"] = result_df.apply(lambda t: calculate_f1_accuracy(t.Actual, t.Model), axis=1)
            em_accuracy = round(sum(result_df["EM"]) / len(result_df), 3)
            f1_accuracy = round(sum(result_df["F1"]) / len(result_df), 3)
            # add to the list
            results.append([set_name, em_accuracy, f1_accuracy])
        return results

    def tabulate_performance(self):
        results = self.evaluate_answers()
        print("CWQ")
        print(tabulate(results, tablefmt="grid", headers=["", "EM", "F1"]))


class FBQA_Dataset(Dataset):
    test_set_path = "/datasets/FreebaseQA/FbQA-eval-1000.csv"
    result_set_paths = [
        "/datasets/FreebaseQA/results/bline.csv",
        "/datasets/FreebaseQA/results/bline2.csv",
        "/datasets/FreebaseQA/results/kb-path.csv",
        *[
            f"/datasets/FreebaseQA/results/kb{d}.csv" 
            for d in range(1, 12)
        ],
        *[
           f"/datasets/FreebaseQA/results/sbert-kb{d}.csv"
           for d in range(1, 12)
        ],
        "/datasets/FreebaseQA/results/tog-lp-4.csv",
    ]

    def __init__(self):
        self.test_set = pd.read_csv(self.test_set_path, index_col=0)
        self.result_df = None

    @property
    def answers(self):
        def extract_answers(row):
            answer_list = []
            parses = ast.literal_eval(row["Parses"])
            for parse in parses:
                for answer in parse.get("Answers", []):
                    answer_list.append(answer["AnswersName"][0])
            return answer_list
        answers = self.test_set.apply(extract_answers, axis=1).reset_index(drop=True)
        return answers

    @property
    def results(self):
        if self.result_df is not None and not self.result_df.empty:
            return self.result_df
        results = [self.answers.tolist()]
        snames = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            snames.append(set_name)
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set)
            # 
            result_df.rename(columns={0: "Model"}, inplace=True)
            results.append(result_df.Model.tolist())
        result_df = pd.DataFrame(results, index=["Actual", *snames]).T
        result_df["quid"] = self.test_set.apply(lambda t: t.get("Question-ID"))
        self.result_df = result_df
        return self.result_df
    
    def evaluate_answers(self):
        results = []
        for r_set in self.result_set_paths:
            set_name = r_set.split("/")[-1].split(".")[0]
            # compute the accuracy on the result set
            result_df = pd.read_csv(r_set, dtype=str)
            result_df.rename(columns={0: "Model"}, inplace=True)
            result_df["Actual"] = self.answers
            result_df["Model"] = result_df["Model"].fillna("").apply(lambda s: s.lower()).apply(lambda t: t.split(","))
            # 
            result_df["EM"] = result_df.apply(lambda t: calculate_em_accuracy(t.Actual, t.Model), axis=1)
            result_df["F1"] = result_df.apply(lambda t: calculate_f1_accuracy(t.Actual, t.Model), axis=1)
            em_accuracy = round(sum(result_df["EM"]) / len(result_df), 3)
            f1_accuracy = round(sum(result_df["F1"]) / len(result_df), 3)
            # add to the list
            results.append([set_name, em_accuracy, f1_accuracy])
        return results
    
    def tabulate_performance(self):
        results = self.evaluate_answers()
        print("FBQA")
        print(tabulate(results, tablefmt="grid", headers=["", "EM", "F1"]))
    
    @staticmethod
    def convert_entity(entity, reverse=False):
        """
        converts: m.07j6w into /m/07j6w
        reverse does the opposite
        """
        if reverse:
            return entity[1:].replace("/", ".")
        return "/" + entity.replace(".", "/")
            

class MetaQA_Dataset(Dataset):
    test_set_path = "/datasets/MetaQA/{hop}/test_1000.txt"
    hops = ["1hop", "2hop", "3hop"]
    result_set_paths = [
        "/datasets/MetaQA/results/{hop}/bline.csv",
        "/datasets/MetaQA/results/{hop}/bline2.csv",
        "/datasets/MetaQA/results/{hop}/kb-path.csv",
        "/datasets/MetaQA/results/{hop}/kb1.csv",
        "/datasets/MetaQA/results/{hop}/kb2.csv",
        "/datasets/MetaQA/results/{hop}/kb3.csv",
        "/datasets/MetaQA/results/{hop}/kb4.csv",
        "/datasets/MetaQA/results/{hop}/sbert-kb1.csv",
        "/datasets/MetaQA/results/{hop}/sbert-kb2.csv",
        "/datasets/MetaQA/results/{hop}/sbert-kb3.csv",
        "/datasets/MetaQA/results/{hop}/sbert-kb4.csv",
        "/datasets/MetaQA/results/{hop}/tog-lp-4.csv",
    ]

    def __init__(self):
        test_sets = {}
        for hop in self.hops:
            tset_path = self.test_set_path.format(hop=hop)
            tset = pd.read_csv(tset_path, header=None, index_col="qid", names=["qid", "Question", "Answers"])

            tset.Answers = tset.apply(lambda t: set(str(t.Answers).lower().split("|")), axis=1)
            test_sets[hop] = tset
        self.test_sets = test_sets
    
    def answers(self, hop):
        return self.test_sets[hop].Answers

    def evaluate_answers(self):
        results = {}
        p = re.compile(r",|\|")
        for hop in self.hops:
            results[hop] = []
            for r_set in self.result_set_paths:
                set_name = r_set.split("/")[-1].split(".")[0]
                # if the file does not exist, 0 accuracy
                if not os.path.exists(r_set.format(hop=hop)):
                    results[hop].append([set_name, 0, 0])
                    continue
                # otherwise, compute the accuracy
                # merge copy of test set with results
                result_df = self.test_sets[hop].copy()
                r_df = pd.read_csv(r_set.format(hop=hop), index_col=0)
                if set_name == "tog-lp-4":
                    r_df = pd.read_csv(r_set.format(hop=hop), index_col=0)
                    r_df.Model = r_df.Model.apply(lambda t: ast.literal_eval(t)[0])
                
                result_df = r_df.merge(result_df, how="left", left_index=True, right_index=True)
                # print(result_df)
                # result_df.insert(2, "Model", r_df.Model)
                # account for empty entries
                result_df.fillna("", inplace=True)
                result_df["Model"] = result_df.apply(lambda t: p.split(str(t.Model)), axis=1).values
                # 
                # print(result_df[["Answers", "Model"]])
                result_df["EM"] = result_df.apply(lambda t: calculate_em_accuracy(t.Answers, t.Model), axis=1)
                result_df["F1"] = result_df.apply(lambda t: calculate_f1_accuracy(t.Answers, t.Model), axis=1)
                em_accuracy = round(sum(result_df["EM"]) / len(result_df), 3)
                f1_accuracy = round(sum(result_df["F1"]) / len(result_df), 3)
                # add to the list
                results[hop].append([set_name, em_accuracy, f1_accuracy])
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
            header.extend([f"{hop}\nEM", f"{hop}\nF1"])
            # method, accuracy
            for i, (method, em, f1) in enumerate(results[hop]):
                table[i].extend([em, f1])

        print("MetaQA")
        print(tabulate(table, tablefmt="grid", headers=header))