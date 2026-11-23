from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from typing import Literal
import cohere

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embedding_model
)

cohore_client = cohere.Client(os.environ["COHERE_API_KEY"])

@tool
def retrieve_relevant_document(
    query: str,
    blog_post: Literal["harness-engineering","diffusion-models"] | None = None
):
    """
        Semantic search over the two posts.
        
        Pass `blog_post` to search inside one blog post: leave it empty to search both
    """
    if blog_post:
        docs = store.similarity_search(query, k=20, filter={"blog_post": blog_post})
    else:
        docs = store.similarity_search(query, k=20)
        
    reranked = cohore_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=[d.page_content for d in docs],
        top_n=5
    )

    return[
        {
            "blog_post": docs[result.index].metadata.get("blog_post",""),
            "text": docs[result.index].page_content
        }
        for result in reranked.results
    ]
    