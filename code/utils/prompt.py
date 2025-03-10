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

A:
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

A:
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

A:
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

RERANK_TRIPLETS_PROMPT = """Rerank the triplets based on how well they help to build a reasoning chain to answer the question and rate their contribution on scale from 0 to 1.
Output in the following format:
1. (relation;score) reason
Only use the relations provided. Do not produce any other text.
Question: {question}
Topic Entity: {entity}
Relations: {relations}
A: 
"""

RERANK_CANDIDATE_ENTITIES_PROMPT = """Rerank the entities based on how well they help to build a reasoning chain to answer the question and rate the contribution on scale from 0 to 1.
Output in the following format:
1. (entity;score) reason
Only use the entities provided. Do not produce any other text.
Question: {question}
Relation: {relation}
Entities: {entities}
A:
"""

PREDICT_TAIL_PROMPT = """
You are an expert in knowledge graphs and natural language understanding. Your task is to help explore relevant relationships from given topic entities that can aid in answering a question.
Instructions:
Input: You will be provided with a natural language question and a list of topic entities extracted from that question.
Objective: Analyze the question to understand its context and what information might be needed to answer it. Then, generate a list of {no_items} candidate relationship labels (i.e., edge types) that could be used to navigate a knowledge graph starting from each entity.
Requirements:
Relevance: The candidate relationship labels must be pertinent to the context of the question.
Conciseness: Provide a brief description (1–2 sentences) of why each relationship label might help answer the question.
Format: Return your answer as a numbered list in the following format: 1. (Entity; Relationship label; Reason)
Do not produce any other text.

Question: “What awards has Albert Einstein received?”
Topic Entities: Albert Einstein;
Candidate relationship labels ({no_items} items):
1. (Albert Einstein; awardReceived; Connects a person to the awards they have received.)
2. (Albert Einstein; honorificAward; Links individuals to awards given in honor of their achievements.)

Question: “{question}”
Topic Entities: {entities}
Candidate relationship labels (5 items):
"""

PREDICT_EDGE_PROMPT = '''
You are an expert in knowledge graphs and natural language understanding. Your task is to help explore relevant relationships from given topic entities that can aid in answering a question.
Instructions:
Input: You will be provided with a natural language question and a list of topic entities extracted from that question.
Objective: Analyze the question to understand its context and what information might be needed to answer it. Then, generate a list of {no_items} candidate relationship labels (i.e., edge types) that could be used to navigate a knowledge graph starting from each entity.
Requirements:
Relevance: The candidate relationship labels must be pertinent to the context of the question.
Conciseness: Provide a brief description (1–2 sentences) of why each relationship label might help answer the question.
Format: Return your answer as a numbered list in the following format: 1. (Entity; Relationship label; Reason)
Do not produce any other text.

Question: “What awards has Albert Einstein received?”
Topic Entities: Albert Einstein;
Candidate relationship labels (2 items):
1. (Albert Einstein; awardReceived; Connects a person to the awards they have received.)
2. (Albert Einstein; honorificAward; Links individuals to awards given in honor of their achievements.)

Question: “{question}”
Topic Entities: {entities}
Candidate relationship labels ({no_items} items):
'''
