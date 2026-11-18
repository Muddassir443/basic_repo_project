# Vector Databases

A vector database stores embeddings and finds the nearest ones to a query vector quickly. For small projects, a NumPy array and a dot product are enough. At larger scale, approximate nearest neighbour indexes such as HNSW trade a little accuracy for much faster search.

Production systems also store metadata next to each vector, such as the source file, page number, or access permissions, so that results can be filtered and cited. Many general purpose databases now offer vector search, which lets teams keep their documents and embeddings in one place.
