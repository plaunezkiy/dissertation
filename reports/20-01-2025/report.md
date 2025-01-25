Credit connectedpapers.com

[hits@k metric](https://stackoverflow.com/questions/58796367/how-is-hitsk-calculated-and-what-does-it-mean-in-the-context-of-link-prediction)

KG-2
Extending the graph traversal to multiple hops away poses an immediate
problem of GPU memory and context windows overflow

To address that, use a reranking mechanism with truncation.
to achieve this, Following ToG, the reranking methods used:
1. BM25 [1]
2. SentenceBERT
3. LLM



References:
1:
@misc{bm25s,
      title={BM25S: Orders of magnitude faster lexical search via eager sparse scoring}, 
      author={Xing Han Lù},
      year={2024},
      eprint={2407.03618},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.03618}, 
}