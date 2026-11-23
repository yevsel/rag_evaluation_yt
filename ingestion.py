import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

BLOG_POST = {
    "harness-engineering": "https://lilianweng.github.io/posts/2026-07-04-harness/",
    "diffusion-models": "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
}

INDEX_NAME = os.environ["INDEX_NAME"]

documents = []

#====
# Scraping
#====

for blog_post, url in BLOG_POST.items():
    for doc in WebBaseLoader(url).load():
        doc.metadata["blog_post"] = blog_post
        doc.metadata["url"] = url
        documents.append(doc)
        

# print("Documents:",documents)


#=====
# Chunking
#=====

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# print("Chunks:",chunks)


#====
# Vector Database
#====

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if pc.has_index(INDEX_NAME):
    pc.Index(INDEX_NAME).delete(delete_all=True)
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    
#=====
# Embedding model
#=====

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME
)