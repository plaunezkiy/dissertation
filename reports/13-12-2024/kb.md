How do I introduce a KB to the model?

1. Train on the KB from scratch
2. Fine-tune a foundation model on the KB
3. Introduce the KB subgraph based on query at inference

Training from scratch (1) is inefficient and would likely overfit,
we would also lose the capabilities of a foundation model

fine tuning results (Unicorn on rainbow) would be interesting 
to compare against at-inference integration

fine-tuning is also resource intensive and not transparent.
Fine-tuning also tends to deprive the model of its 
generalized capabilites by fitting strongly to the training set

I am aiming for a transparent reasoning chain 
(highlight relevant subraphs that were used)

Big question is What is a good/relevant KB
gotta look into how they are generated

KnowledgeNET is an open-source baseline that has human annotated
triplets based on natural text. Derived from Wikidata

I wanna study the KB population in more detail

Get a dataset, try on a plain model, see the accuracy,
look into attention masks (maybe something interesting there)


Vocab growth follows a power law [Heap's law](https://en.wikipedia.org/wiki/Heaps%27_law)
get a bunch of text corpora and start processing
them to extract relations.
There will be a ton of new nodes and edges

Different Nodes can refer to the same entity, so Id need
to perform some form of coreference resolution to link
entities.
This can be done using embeddings similarity with a
cutoff point? This is where Heap's law could come in handy.
I can empirically fit the parameter of the similarity
threshold to match the vocab growth function

I can use a bootstrap algorithm to extract everything I can
and somehow qunatify the confidence of the relations.

Computing embeddings on Mistral's first layer might not be the 
best idea as M7B is a decoder-only model (predict left-to-right)
and therefore the initial layer of embeddings is theoretically 
only good for next-word prediction. Due to unidirectional 
training objective embeddings may lack rich linguistic 
information that could potentially be useful for coreference resolution

I need an encoder-only/encoder-decoder model for this

Immediate TODO (by next week):
- Introduce FB
- Get few-shot results (combine or pick one, or average)

Entity search by similarity (rather than the exact match)

Potentially:
- skip Cosmos KB as it is not clear how to integrate (subj to research)
- preselect the nodes (entities)
- consider an interface to interact with a document 
(links and hierarchy between sections of a document)

Pick something niche and figure it out to make it better

for a superspecific problem, a specific problem might have a solution
referenced arbitrarily far away. 
