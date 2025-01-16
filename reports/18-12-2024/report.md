Created a custom LLM class for local Mistral
inference chains with LangChain

set up KB integration by using LangChain
GraphQAChain that extracts entities from a query,
converts them to KB nodes. Each node's
triplets are extracted, combined into a context string
loaded with a question and sent off to the LLM.

The first big issue is QN=84, many entities with 
many edges, so all of the triplets take over all free GPU memeory, causing CUDA Out Of Memory.

Need some form of reranking and thresholding.
For now, just truncate the input. Not ideal,
but I wouldn't expect many cases of ^ to occur

got over 800 triplets, I empirically limited
it to 600 to prevent OutOfMem errors
Should be limited by tokens, not numbers!!
Relevance should also affect which triplets are skipped.