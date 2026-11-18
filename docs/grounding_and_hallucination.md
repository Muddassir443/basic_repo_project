# Grounding and Hallucination

A language model can produce fluent answers that are wrong, which is called hallucination. Grounding reduces this by instructing the model to answer only from retrieved context and to say it does not know when the context lacks the answer. Asking the model to cite the chunk it used makes each claim traceable to a source.

Quality of retrieval limits quality of the answer. If the right passage is not retrieved, the model cannot use it. Teams measure retrieval with metrics such as recall at k, which asks whether the correct passage appears in the top k results.
