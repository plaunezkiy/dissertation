from langchain_core.prompts.prompt import PromptTemplate

EXTRACT_ENTITIES_TEMPLATE = """
Below is a question, extract all entities and return them separeted by a coma.
Only return the entities with no extra text and separators. 
Do not introduce or deduce anything not present in the question.
Question:
{question}"""

ENTITY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=EXTRACT_ENTITIES_TEMPLATE
)

LEN_LIMITED_PROMPT = """
You will be given a context containing relevant entity triplets and a question.
Using the context, you should answer the question below in under a sentence 
with no other text but the answer.
Do not include any additional text.
Separate individual answers by a coma.
Context:
{context}

Question:
{question}
"""

GRAPH_QA_PROMPT = PromptTemplate(
    template=LEN_LIMITED_PROMPT, input_variables=["context", "question"]
)

LEN_LIMITED_PROMPT_NO_CONTEXT = """
You should answer the question below in under a sentence 
with no other text but the answer.
Do not include any additional text.
Separate individual answers by a coma.
Question:
{question}
"""

NO_CONTEXT_PROMPT = PromptTemplate(
    template=LEN_LIMITED_PROMPT_NO_CONTEXT, input_variables=["question"]
)

EVALUATE_CONTEXT = """
You will be given a context containing relevant entity triplets and a question.
Using this, you should determine if the provided context is enough to
answer the question.
YES if the context is sufficient, NO if it is not sufficient.
Do not include any additional text.
Only output YES or NO.
Context:
{context}

Question:
{question}
"""

EVALUATE_CONTEXT_PROMPT = PromptTemplate(
    template=EVALUATE_CONTEXT, input_variables=["context", "question"]
)