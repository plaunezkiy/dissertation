class QAItem:
    class QParse:
        class Answer:
            def __init__(self, obj):
                self.mid = obj["AnswersMid"]
                self.name = obj["AnswersName"]
        
        def __init__(self, obj):
            self.ID = obj["Parse-Id"]
            self.chain = obj["InferentialChain"]
            self.answers = [self.Answer(o) for o in obj["Answers"]]

    def __init__(self, obj):
        self.ID = obj["Question-ID"]
        self.question = obj["RawQuestion"]
        self.parses = [self.QParse(o) for o in obj["Parses"]]
