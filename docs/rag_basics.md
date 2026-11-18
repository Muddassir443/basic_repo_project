# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a search step with a language model. Instead of relying only on what the model memorised during training, the system first retrieves relevant passages from a document collection and passes them to the model as context. This makes answers more grounded, easier to verify, and possible to update without retraining the model.

A typical RAG pipeline has two phases. During ingestion, documents are split into chunks, each chunk is converted into an embedding vector, and the vectors are stored in an index. During querying, the question is embedded the same way, the most similar chunks are retrieved, and the language model writes an answer using only those chunks.
