from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd 

df = pd.read_csv("knowledge_base.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

add_document = not os.path.exists(db_location)

if add_document:
    documents = []
    ids = []
    for index, row in df.iterrows():
        document = Document(
            page_content=str(row["id"]) + " " + row["text"], 
            metadata={"source": row["source"]}
        )
        documents.append(document)
        ids.append(str(row["id"]))

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="reviews",
        persist_directory=db_location,
        ids=ids
    )
else:
    vector_store = Chroma(
        persist_directory=db_location,
        embedding_function=embeddings,
        collection_name="reviews"
    )

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
) 


