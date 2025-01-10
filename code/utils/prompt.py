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
Using the context, you should answer the question below in under a sentence with no other text but the answer.
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
