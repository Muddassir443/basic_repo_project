# Chunking and Embeddings

Chunking means splitting long documents into smaller pieces so that retrieval returns focused passages rather than entire files. Chunks that are too large dilute the meaning and waste the model's context window. Chunks that are too small lose surrounding information. Overlap between neighbouring chunks helps keep sentences that straddle a boundary from being lost.

An embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar meaning end up close together in vector space, even if they share no words. Semantic search measures closeness with cosine similarity, which compares the angle between two vectors. If the vectors are normalised to length one, cosine similarity is simply their dot product.
