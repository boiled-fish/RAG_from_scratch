import requests
from langchain.llms.ollama import Ollama
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from datasets import load_dataset
from tqdm import tqdm

class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'mxbai-embed-large:latest', base_url='http://localhost:11434'):
        self.model_name = model_name
        self.base_url = base_url

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
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
    def __init__(self, k=5, generator_model='llama3:latest'):
        self.k = k
        self.generator_model = generator_model
        print(f"Generator model: {self.generator_model}")
        self.embedding = self._get_embedding()
        print(f"Embedding: {self.embedding}")
        self.vector_store = self._build_vector_store()
        print(f"Vector store: {self.vector_store}")
        self.retriever = self._get_retriever()
        print(f"Retriever: {self.retriever}")
        self.generator = self._get_generator()
        print(f"Generator: {self.generator}")

    def _get_embedding(self):
        # Initialize the embedding model
        embedding = OllamaEmbeddings(model_name='mxbai-embed-large:latest')
        return embedding

    def _build_vector_store(self):
        # Load the WikiQA dataset
        qa_dataset = load_dataset(
            'rag-datasets/rag-mini-wikipedia', 
            'question-answer',
            cache_dir='./datasets'
        )

        corpus_dataset = load_dataset(
            'rag-datasets/rag-mini-wikipedia',
            'text-corpus',
            cache_dir='./datasets'
        )

        # Extract the test split for QA data
        qa_data = qa_dataset['test']

        # Create Document objects from the corpus and QA dataset
        documents = [Document(page_content=entry['passage']) for entry in corpus_dataset['passages']]
        for item in tqdm(qa_data, desc="Building documents"):
            question = item['question']
            answer = item['answer']
            text = f"Question: {question}\nAnswer: {answer}"
            documents.append(Document(page_content=text))

        # Split documents into chunks
        print("Starting to split documents")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Build the vector store with embeddings
        print("Starting to build vector store")
        vector_store = FAISS.from_documents(docs, self.embedding)
        print("Vector store built")
        return vector_store

    def _get_retriever(self):
        # Create a retriever from the vector store
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
        return retriever

    def _get_generator(self):
        # Select the generator model based on the provided model name
        if self.generator_model in ['llama3:latest', 'gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini']:
            # Use Ollama for these models
            generator = Ollama(model=self.generator_model)
        elif self.generator_model in ['openai-gpt-4', 'openai-gpt-3.5-turbo']:
            # Use OpenAI API for these models
            generator = OpenAI(model_name=self.generator_model.replace('openai-', ''))
        else:
            raise ValueError(f"Unsupported generator model: {self.generator_model}")
        return generator

    def generate(self, query):
        # Create a RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=self.generator,
            retriever=self.retriever,
            return_source_documents=True
        )
        # Generate the answer
        result = qa({"query": query})
        return result

    def pipeline(self, query):
        # Orchestrate the process
        return self.generate(query)

    def add_documents(self, new_documents):
        """
        Adds new documents to the corpus and updates the vector store.

        Args:
            new_documents (List[str] or List[Document]): The new documents to add.
        """
        # Convert strings to Document objects if necessary
        if all(isinstance(doc, str) for doc in new_documents):
            new_documents = [Document(page_content=text) for text in new_documents]
        elif all(isinstance(doc, Document) for doc in new_documents):
            pass
        else:
            raise ValueError("new_documents must be a list of strings or Document objects")

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(new_documents)

        # Add documents to the vector store
        self.vector_store.add_documents(docs)
