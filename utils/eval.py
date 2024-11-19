import csv


class QADataset:
    questions: dict
    answer_choices: dict
    answers: dict
    recorded_responses: dict

    def __init__(self):
        self.questions = {}
        self.answer_choices = {}
        self.answers = {}
    
    def load(self):
        pass


class CosmosQA(QADataset):
    path = "../\datasets\CosmosQA\valid.csv"
    
    def load(self):
        with open(self.path) as ds_file:
            reader = csv.reader(ds_file)
            # Skip header
            next(reader)
            for row in reader:
                qid, context, question, answer0, answer1, answer2, answer3, label = row
                self.questions[qid] = question
                self.answer_choices[qid] = [answer0, answer1, answer2, answer3]
                self.answers[qid] = label

    def evaluate(self, model):
        for qid, question in self.questions.items():
            answer = self.answers[qid]
            options = self.answer_choices[qid]
            response = model.predict(question, options)
            self.recorded_responses[qid] = response
            if response == answer:
                print(f"Correct: {question} -> {response}")


class QAEvaluator:
    def __init__(self):
        pass