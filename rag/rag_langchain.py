import os
import requests
from langchain.llms.ollama import Ollama
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeVectorStore  # Official LangChain Pinecone
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec, PodSpec
import logging
from typing import List, Union
import torch
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEmbeddings(Embeddings):
    def __init__(self, 
                 model_name: str = 'mxbai-embed-large:latest', 
                 base_url='http://localhost:11434', 
                 batch_size: int = 32):
        self.model_name = model_name
        self.base_url = base_url
        self.batch_size = batch_size
        self.device = torch.device("cpu")  # Ensure embeddings are processed on CPU

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                batch_embeddings = list(executor.map(self.embed_query, batch))
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            embedding = data.get('embedding', [])
            return embedding
        else:
            raise Exception(f"Failed to get embedding: {response.status_code}, {response.text}")

class RAGModel:
    def __init__(self, 
                 k: int = 5, 
                 generator_model: str = 'llama3:latest', 
                 cache_dir: str = './cache',
                 pinecone_api_key: str = 'YOUR_PINECONE_API_KEY',  # Replace with your Pinecone API key
                 pinecone_env: str = 'YOUR_PINECONE_ENVIRONMENT'  # Replace with your Pinecone environment
                ):
        self.k = k
        self.generator_model = generator_model
        self.cache_dir = cache_dir
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.embedding = self._get_embedding()
        self.pinecone_client = self._init_pinecone()
        self.vector_store = self._build_vector_store()
        self.retriever = self._get_retriever()
        self.generator = self._get_generator()

    def _get_embedding(self):
        return OllamaEmbeddings(
            model_name='mxbai-embed-large:latest',
            batch_size=32
        )

    def _init_pinecone(self):
        # Initialize Pinecone client
        pinecone_client = PineconeClient(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_env
        )
        return pinecone_client

    def _build_vector_store(self):
        # Define Pinecone index name
        index_name = "rag-index"

        # Create index if it does not exist
        if index_name not in self.pinecone_client.list_indexes():
            logger.info(f"Creating Pinecone index '{index_name}'...")
            self.pinecone_client.create_index(
                name=index_name,
                dimension=768,  # Adjust based on your embedding dimension
                metric="euclidean",
                spec=PodSpec(pod_type="p1")  # Correctly specify the pod type using PodSpec
            )
            logger.info(f"Pinecone index '{index_name}' created.")
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")

        # Connect to the Pinecone index
        index = self.pinecone_client.Index(index_name)

        # Load datasets
        qa_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'question-answer', cache_dir='./datasets')
        corpus_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'text-corpus', cache_dir='./datasets')

        # Prepare documents
        documents = []

        # Process corpus dataset
        corpus_docs = [Document(page_content=entry['passage']) for entry in corpus_dataset['passages']]
        documents.extend(corpus_docs)

        # Process QA dataset
        qa_data = qa_dataset['test']
        qa_docs = []
        for item in tqdm(qa_data, desc="Building QA documents"):
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            qa_docs.append(Document(page_content=text))
        documents.extend(qa_docs)

        # Split documents
        logger.info("Splitting documents...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create embeddings in batches
        logger.info("Creating embeddings with CPU processing...")
        texts = [doc.page_content for doc in docs]
        embeddings = []

        batch_size = 1000  # Adjust as needed
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

        # Upsert to Pinecone
        logger.info("Upserting embeddings to Pinecone...")
        ids = [f"doc_{i}" for i in range(len(texts))]
        vectors = embeddings
        index.upsert(vectors=list(zip(ids, vectors)))
        logger.info("Embeddings upserted to Pinecone.")

        # Initialize LangChain Pinecone vector store
        vector_store = PineconeVectorStore(
            embedding_function=self.embedding.embed_query,  # Ensure this matches your embedding setup
            index=index
        )

        return vector_store

    def _get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": self.k})

    def _get_generator(self):
        if self.generator_model in ['llama3:latest', 'gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini']:
            return Ollama(model=self.generator_model)
        elif self.generator_model in ['openai-gpt-4', 'openai-gpt-3.5-turbo']:
            return OpenAI(model_name=self.generator_model.replace('openai-', ''))
        else:
            raise ValueError(f"Unsupported generator model: {self.generator_model}")

    def generate(self, query: str) -> dict:
        qa = RetrievalQA.from_chain_type(
            llm=self.generator,
            retriever=self.retriever,
            return_source_documents=True
        )
        return qa({"query": query})

    def pipeline(self, query: str) -> dict:
        return self.generate(query)

    def add_documents(self, new_documents: List[Union[str, Document]], batch_size: int = 32):
        if all(isinstance(doc, str) for doc in new_documents):
            new_documents = [Document(page_content=text) for text in new_documents]
        elif not all(isinstance(doc, Document) for doc in new_documents):
            raise ValueError("new_documents must be a list of strings or Document objects")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(new_documents)

        # Embed and upsert new documents to Pinecone
        logger.info("Adding new documents to Pinecone vector store...")
        texts = [doc.page_content for doc in docs]
        embeddings = self.embedding.embed_documents(texts)
        ids = [f"doc_new_{i}" for i in range(len(texts))]  # Generate unique IDs
        vectors = embeddings
        self.pinecone_client.Index("rag-index").upsert(vectors=list(zip(ids, vectors)))
        logger.info("New documents upserted to Pinecone.")

    def change_generator(self, new_generator: str):
        self.generator = self._get_generator(new_generator)

if __name__ == "__main__":
    # Initialize with Pinecone vector store
    rag = RAGModel(
        pinecone_api_key="4acd1cb7-e133-4cfb-a5fc-1267e5f91515",  # Ensure this API key is correct
        pinecone_env=os.getenv('PINECONE_ENVIRONMENT')  # Ensure this environment variable is set
    )
    result = rag.generate("What is the capital of France?")
    print(result)
