
Created: 17-01-2026 15:25

Status: #baby 

Tags: [[retrieval-augmented-generation]] [[hybrid-memory]] [[natural-language-processing]] [[rag-pipeline-implementation]] [[vector-databases-in-rag]] [[gpt-2-characteristics]] [[attention-mechanisms-in-transformers]] [[agentic-rag-approach]]

# Retrieval-Augmented Generation

## Definition
Retrieval-Augmented Generation (RAG) is a model architecture that combines parametric memory (knowledge encoded in neural network weights) with non-parametric memory (external knowledge retrieved from a document database) to enhance the performance of natural language processing tasks. This approach aims to provide accurate and contextually relevant information, thereby addressing limitations found in models solely reliant on either type of memory.

## How It Works
RAG utilizes two components: a retriever and a generator. The retriever identifies relevant documents from a database using a bi-encoder architecture, which encodes both the queries and documents to find similarities. In contrast, the generator similarly processes this external context along with the query to generate outputs. The model can operate with two different objectives: RAG-Sequence uses the same document for predicting the entire output sequence, while RAG-Token generates each token potentially based on different documents, allowing for more flexibility and nuanced responses.

## Key Points
- **Hybrid Memory**: Combines parametric (learned during training) and non-parametric (fetched from a document index) approaches to access knowledge.
- **Two Modes**: Implements RAG-Sequence for single-document dependency and RAG-Token for multiple documents influencing each token generated.
- **Document Indexing**: The model utilizes a dense passage retriever (DPR) for encoding documents and queries into dense vectors, facilitating efficient retrieval processes.
- **Marginalization Process**: Marginalizes over output sequences or tokens to determine the most probable result, enhancing accuracy based on document relevance scores.
- **Task Flexibility**: RAG shows versatility across various knowledge-intensive tasks like question answering and fact verification, outperforming traditional parametric models.
- **Real-Time Knowledge Updating**: The model can update its knowledge dynamically by swapping out its document corpus without requiring retraining, making it adaptable to current events and information.

## Connections
RAG builds on previous frameworks like the Dense Passage Retriever and BART transformer models, enhancing their capabilities through integrated retrieval.
# References

1. [[RAG for NLP Tasks]]