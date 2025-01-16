Project structure:
1. Plain LLM QA ability (baseline performance)
2. KG integration performance
3. KG creation (semi-supervised)
4. Document partition (figure, table, paragraph + reference extraction) - metadata generation
5. QA interface for engineers
6. Engineer usage survey (improvement)

TODO:
apply visibility mask to extracted entities and triplets.
This neat trick should potentially alleviate context 
oversaturation issue.


Interface sequence:
1. Extract annotations (figures, tables, paragraphs)
2. for textual elements, extract references

### Project
[1]:
The initial goal of the project was to improve the ability of LLMs to process tabular data embedded in PDFs related to Control Valves.
That is, a specific case of domain-specific multi-modal reasoning. 
[2]:
After extensive research into table-based reasoning, I found
the task quite broad to fit into an honours project, given that
it is both a university and a company-sponsored project.

Given the flexibility I was provided with, I thought that 
focusing on the bigger picture of the project is more
interesting and is a place where I could contribute more.
[3]:
Thus, I have gone deeper into the project to understand
what the requirements on the company side were to try
and innovate there.
The goal of the project is to address the issue of
scarcity of experienced engineers and help less experienced 
ones to navigate complex technical documentation related
to control valves when it comes to handling equipment
such as control valves used at industrial plant sites.
[4]:
I was provided with a collection of such documents, as well as 
a list of sample queries an engineer might have. 
The immediate problem I faced was evaluation criteria.
Given a query and an answer, how do I determine the correctness?
For the sample queries, there were no sample answers.
Since the application is domain-specific, for research purposes
I selected additional domains to work on and show
cross-domain applications of my ideas.

[5]:
Due to scarcity of testing data, I decided to use available
open-source domains with corresponding KBs. This also helps
to show the inter-domain applications of my methodology.

When selecting other domain, the key requirements were:
1. QA dataset with correct answers
2. A corresponding KB

The ones that fit the requirements were:
1. FBQA - trivia-style questions about general knowledge facts (web based)
2. MetaQA - movie-based questions (actors, directors, dates)
3. CosmosQA - common-sense reasonoing

(mention sizes, data types, corresponding KB format,o #qs)

[6]:
### Methodology

The high-level approach is as such:
Use a knowledge base (text, images, tables) related to something
specific to answer questions.
This can be broken down into standalone tasks:
- Preprocessing KB for reasoning
- Processing the query
- Generating enriched answer
- Evaluating the answer

With this, it was time to establish wheteher using a KB is
even feasible and if it gives any improvement.
The baseline to compare is plain LLM prompting with no
additional information. This relies on training data
the LLM was provided with.

[7-8]:
Current situtation + plan

## Progress

### Model selection

Given the potential commerical application of the project,
this restricted the range of models I could use. Provided
with limited hardware, the range narrowed down more
Finally, Mistral7B was chosen as a reasoning agent for 
this project. The framework developed is model agnostic, however.

[9]:
### Dataset analysis

Looking at the format of the answers in the QA dataset,
they are names, entities, locations, in other words, single
words or sentences.
Before running experiments at scale, after experimenting, it was
noted that the LLM, due to how it was trained, tends to produce 
additional text such as "Sure, here's what I know about ...".
To account for this and minimize extra text, I empirically
found the prompt that produces a single answer to the question.

[10]:
### Baseline

To evaluate the correctness in natural text (machine translation/summarization),
researches usually use some form of recall/precision-based metrics (BLEU, ROUGE).
The difference is in preference of generated answer proportion of matching text,
or the proportion of the correct answer.
For this project, I made certain assumptions here. Since the domain-specific
application is for educated engineers, a proportion of the correct answer is 
in favour here as a clued-in individual is likely to make sense of the answer
if it has the correct terms in it. However, overly long text may confuse
the reader, thus I decided to use a recall-based accuracy score to evaluate
generated answers. to penalize long answers, I will use exponential penalty
score that penalises overly long answers.

With this established, I ran my first set of experiments to establish the 
baseline performance and obtained the following results:
FBQA
+----------+------------+
| Method   |   Test Set |
+==========+============+
| bline    |   0.736486 |
+----------+------------+
| bline2   |   0.737738 |

MetaQA
+----------+----------+----------+----------+
| Method   |     1hop |     2hop |     3hop |
+==========+==========+==========+==========+
| bline    | 0.390168 | 0.231509 | 0.418103 |
+----------+----------+----------+----------+
| bline2   | 0.394591 | 0.232316 | 0.415861 |

CosmosQA
+==========+==========+
| bline    | 0.677    |
+----------+----------+
| bline2   | 0.655    |

I ran the baseline twice to make sure the LLM has a 
consistent performance and the results show it does so.

### KB
[11]:
to introduce the knowledge, several approaches were explored:
1. Training the model from scratch on the KB
2. Fine-tuning a foundation model on the KB
3. Introducing KB at inference time

1 is resource intensive and requires a lot of quality data - not feasible,
2 is also resource intensive, and also damages the generalizability of the model
by overfitting to the tuning set
3 is the preferred option as it is model agnostic and relatively cheap.
I am also aiming for a transparent reasoning chain to make the process clear for engineers,
so 1 and 2 are even less favoured.

[12]:
The kb came as bits of text with corresponding text and extracted relation triplets.
The 2 ways of introducing it are:
1. RAG - text chunks
2. Triplet insertion

since the text is unstructured and is (atm) not feasible to work with due to its size.
I worked with triplets. Similarly, by empirically selecting a prompt and introducing 
context as well as the question I ran the experiments and obtained the following results:
[13]:
FBQA KB
+----------+------------+
| kb1      |   0.635886 |
+----------+------------+
MetaQA
+----------+----------+----------+----------+
| kb1      | 0.862371 | 0.175632 | 0.34202  |
+----------+----------+----------+----------+

CosmosQA has Atomic as a corresponding KB, but it comes as a sequence of
common-sense sequences (do A to get B) and it was unclear how to integrate
that into the process (abstract concepts) and thus it was decided to move
on without Cosmos for now

### Intermediate results

The results show that given a knowledge base, the trend emerges. With fbqa the accuracy
decreased (show multihop reasoning). For metaqa, it came partitioned into hops (nodes away)
and since the approach only introduced knowledge of the first order neighbours and eges,
the performance increased for 1-hop and decreased for the rest. Upon examining FBQA
structure, the accuracy decreased with the number of hops.
This goes in line with my evaluation assumption that oversaturated text confuses the agent
and produces an even worse answer

Feasibility of using KB to answer questions is there, but necessitates that the answer
is present in the context. 
This calls for a good retrieval strategy.
For most approaches, RAG, KG, text search, the search system and the reasoning agent
are treated as a static black-box that only takes in a prompt and sometimes a 
document. 
Here, I decided to develop a heuristic that uses a hybrid approach and allows the user
to interact with the document and help the agent help the user.

Turning to the technical documentation, it is immediately clear that the document has
many different elements (text, figures, tables, headers, footer). Feeding it directly
to the model is prone to afforementioned oversaturation issue, so ammendements are in order.

Text and figure annotations also have references to other sections and figures, and extracting
those is integral to generating a comprehensive answer.


### Document interface
The current stage I'm at is preprocessing the document to generate a KB in the form
of metadata that's compatible with the reasoning framework and user interactions.
Should look like (sshot) annotated sections and a chat.
References and entities should be extracted. 
Generated answers should include and highlight those references and elements for full
transparency.

### Document preprocessing
Part of the goal is to generate the metadata. This requires classification/partition
as chunks may be on consecutive pages (remove headers), etc.
This is the next step. Either train a classifier or simply use the LLM itself to
generate the splits.
Dataset:
text+tables+figures (wikipedia + documents), hand annotated. Manually or LLM generated
50/50 split to see if anything interesting happens.

### Engineer survey
The final step is to assess whether the interface is, and how (un)helpful it is.
To do so, run a survey or build a feedback tool into the interface
Measurables:
- Accuracy (1-5 score). Does the LLM provide correct/useful answers.
- Usability (1-5 score). Is the interface intuitive and efficient to use.
- Usefullness (1-5)
- Open ended feedback - free style text answer for feedback and suggestions
