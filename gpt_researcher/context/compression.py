import os
from .retriever import SearchAPIRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gpt_researcher.utils.costs import estimate_embedding_cost
from gpt_researcher.memory.embeddings import OPENAI_EMBEDDING_MODEL
from tiktoken import encoding_for_model

def filter_docs_to_token_limit(docs, max_tokens=250000, model=OPENAI_EMBEDDING_MODEL):
    """Reduce document list to stay under token limits. Works with dicts or Document objects."""
    enc = encoding_for_model(model)
    filtered = []
    total_tokens = 0
    for doc in docs:
        if isinstance(doc, dict):
            text = doc.get("page_content", "")
        else:
            text = getattr(doc, "page_content", "")
        tokens = len(enc.encode(text))
        if total_tokens + tokens > max_tokens:
            break
        filtered.append(doc)
        total_tokens += tokens
    print(f"[DEBUG] Filtered docs count: {len(filtered)}, total tokens: {total_tokens}")
    return filtered


class ContextCompressor:
    def __init__(self, documents, embeddings, max_results=5, **kwargs):
        self.max_results = max_results
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.38))

        # Filter documents to stay under embedding token limit
        self.documents = filter_docs_to_token_limit(documents)
        if not self.documents:
            print("[WARN] No relevant context found — skipped due to empty or over-limit input.")

    def __get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
        
        def limit_chunks(chunks, max_tokens=280000):
            enc = encoding_for_model(OPENAI_EMBEDDING_MODEL)
            filtered = []
            total_tokens = 0
            for chunk in chunks:
                text = chunk.page_content
                tokens = len(enc.encode(text))
                if total_tokens + tokens > max_tokens:
                    break
                filtered.append(chunk)
                total_tokens += tokens
            print(f"[DEBUG] Filtered chunks count: {len(filtered)}, total tokens: {total_tokens}")
            return filtered

        class TokenLimitEmbeddingsFilter(EmbeddingsFilter):
            def compress_documents(self, documents, query=None, **kwargs):
                limited_docs = limit_chunks(documents)
                return super().compress_documents(limited_docs, query=query, **kwargs)

        relevance_filter = TokenLimitEmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold
        )
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        
        base_retriever = SearchAPIRetriever(pages=self.documents)
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        
        return contextual_retriever

    def __pretty_print_docs(self, docs, top_n):
        return "\n".join(
            f"Source: {d.metadata.get('source')}\n"
            f"Title: {d.metadata.get('title')}\n"
            f"Content: {d.page_content}\n"
            for i, d in enumerate(docs) if i < top_n
        )

    def get_context(self, query, max_results=5, cost_callback=None):
        compressed_docs = self.__get_contextual_retriever()
        if cost_callback:
            cost_callback(
                estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents)
            )
        relevant_docs = compressed_docs.invoke(query)
        return self.__pretty_print_docs(relevant_docs, max_results)