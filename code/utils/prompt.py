from langchain_core.prompts.prompt import PromptTemplate

EXTRACT_ENTITIES_TEMPLATE = """
Below is a question, extract all entities and return them separeted by a coma.
The entities should be relevant for traversing a knolwedge graph to 
answer the question.
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

EXTRACT_RELATION_PROMPT = """Retrieve {num_rel} relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {num_rel} relations is 1).
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: {question}
Topic Entity: {entity}
Relations: {relations}
A: 
"""

RERANK_TRIPLETS_PROMPT = """Rerank the triplets based on how well they help to build a reasoning chain to answer the question and rate their contribution from on scale from 0 to 1.
Output in the following format:
1. (relation;score) reason
Only use the relations provided. Do not produce any other text.
Question: {question}
Topic Entity: {entity}
Relations: {relations}
A: 
"""