import os
import json
import glob
from typing import List, Optional
import PyPDF2
import re
import magic

from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chat_models import ChatOpenAI  # Assuming ChatGPT-4 via OpenAI

import logging
import sys

from langchain.schema import Document

# Enhanced CustomLogger with timestamp and better formatting
class CustomLogger:
    def __init__(self, log_dir_base, logger_name):
        self.log_dir_base = log_dir_base
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)

        self.log_dir = f"{log_dir_base}/{logger_name}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Prevent adding multiple handlers if logger is already configured
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

            # Create file handler
            fh = logging.FileHandler(f"{self.log_dir}/{logger_name}.log")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Create console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger

    def get_log_dir(self):
        return self.log_dir

def PDFLoader(file_path: str):
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                if page.extract_text():
                    text += page.extract_text() + " "
            
            # Normalize whitespace and clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split text into chunks by sentences, respecting a maximum chunk size
            sentences = re.split(r'(?<=[.!?]) +', text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < 1000:
                    current_chunk += (sentence + " ").strip()
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk)
            return chunks
    except Exception as e:
        print(f"[ERROR] Error processing PDF file: {e}")
        return []
    
def load_documents(path):
    """
    Recursively load all documents with enhanced metadata.
    """
    documents = []
    supported_extensions = {'.txt', '.md', '.pdf'}
    
    try:
        print(f"[INFO] Starting to scan directory: {path}")
        if not os.path.exists(path):
            print(f"[ERROR] Directory does not exist: {path}")
            return []
            
        for root, dirs, files in os.walk(path):
            print(f"[INFO] Scanning directory: {root}")
            print(f"[INFO] Found {len(files)} files")
            
            # Get relative path for better context
            rel_path = os.path.relpath(root, path)
            folder_structure = rel_path.split(os.sep)
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext not in supported_extensions:
                    continue
                
                try:
                    # Enhanced metadata
                    metadata = {
                        "source": file_path,
                        "file_name": file,
                        "file_type": ext[1:],
                        "folder_path": rel_path,
                        "folder_name": os.path.basename(root),
                        "folder_structure": folder_structure,
                        "full_path": file_path
                    }
                    
                    if ext in {'.txt', '.md'}:
                        try:
                            loader = TextLoader(file_path, encoding='utf-8')
                            docs = loader.load()
                            # Add metadata to each document
                            for doc in docs:
                                doc.metadata.update(metadata)
                            documents.extend(docs)
                        except UnicodeDecodeError:
                            loader = TextLoader(file_path, encoding='gbk')
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata.update(metadata)
                            documents.extend(docs)
                    
                    elif ext == '.pdf':
                        chunks = PDFLoader(file_path)
                        if chunks:
                            from langchain.schema import Document
                            docs = [Document(
                                page_content=chunk, 
                                metadata=metadata
                            ) for chunk in chunks]
                            documents.extend(docs)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {str(e)}")
                    continue
        
        return documents
    
    except Exception as e:
        print(f"[ERROR] Error walking through directory {path}: {str(e)}")
        return []

class RAGModel:
    def __init__(self, model_type: str, note_folder_path: str, top_k: int = 5):
        """
        Initialize the RAG system.
        """
        # Initialize the CustomLogger
        self.logger = CustomLogger(log_dir_base='./log', logger_name="langchain_rag")
        self.log_dir = self.logger.get_log_dir()
        self.log = self.logger.get_logger()

        # Set environment variables for LangChain
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "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"
        os.environ["LANGCHAIN_PROJECT"] = "ECE_RAG"

        self.model_type = model_type.lower()
        self.note_folder_path = note_folder_path
        self.conversation_history = []
        self.top_k = 5

        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        self.log.info("Initialized embedding model")

        # Load and process documents
        self.log.info(f"Loading documents from {note_folder_path}")
        documents = load_documents(self.note_folder_path)
        
        # Split documents with metadata preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        splits = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            # Create new documents with preserved metadata
            splits.extend([
                Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ) for chunk in chunks
            ])
        
        self.log.info(f"Created {len(splits)} text chunks")
        
        # Initialize vector store with enhanced metadata
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_model,
            collection_name="notes_collection",
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.log.info("Initialized vector store")
        
        # Initialize the appropriate model
        self._initialize_generator_model()

    def _initialize_generator_model(self):
        """
        Initialize the generator model based on the current model_type.
        """
        if self.model_type == "chatgpt-4":
            self.generator = ChatOpenAI(model_name="gpt-4", temperature=0.7)
            self.log.info("Initialized ChatGPT-4 model.")
        elif self.model_type == "ollama":
            self.generator = ChatOllama(model="llama3:latest")
            self.log.info("Initialized Ollama model.")
        else:
            self.log.error("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")
            raise ValueError("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")

    def switch_model(self, new_model_type: str):
        """
        Switch the generator model to a different type.

        :param new_model_type: The new model type ('ChatGPT-4' or 'ollama').
        """
        if new_model_type.lower() not in ["ChatGPT-4", "Ollama"]:
            self.log.error("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")
            return False
        
        self.model_type = new_model_type.lower()
        self._initialize_generator_model()
        self.log.info(f"Switched to model: {self.model_type}")

        return True

    def rewrite_query(self, user_input_json: str) -> str:
        """
        Rewrite the user query by incorporating conversation history.

        :param user_input_json: JSON string containing the user's query.
        :param conversation_history: List of [sender, message] pairs.
        :return: Rewritten query string.
        """
        user_input = json.loads(user_input_json).get("Query", "")
        # Get the last two messages from conversation history
        context_messages = self.conversation_history[-2:] if len(self.conversation_history) >= 2 else self.conversation_history
        context = "\n".join([f"{msg[0]}: {msg[1]}" for msg in context_messages])
        prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
The rewritten query should:

- Preserve the core intent and meaning of the original query
- Expand and clarify the query to make it more specific and informative for retrieving relevant context
- Avoid introducing new topics or queries that deviate from the original query
- DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

Return ONLY the rewritten query text, without any additional formatting or explanations.

Conversation History:
{context}

Original query: [{user_input}]

Rewritten query:
"""
        # Use the generator to get the rewritten query
        if self.model_type == "ollama":
            response = self.generator.invoke(prompt)
        else:
            response = self.generator(prompt)

        # Extract the text from the AIMessage object
        rewritten_query = response.content.strip()  # Assuming 'content' is the attribute holding the text
        self.log.info(f"Rewritten query: {rewritten_query}")

        return rewritten_query

    def answer_query(self, query: str, attached_files: List[dict] = []) -> str:
        """
        Enhanced answer generation with better context understanding.
        """
        if attached_files:
            query = f"\n\nOriginal Query: {query}\n\nAttached Files:\n\n"
            for file in attached_files:
                content = "\n".join(file['file_content'])
                query += f"{file['file_name']}:\n{content}\n\n"
        
        self.conversation_history.append(["User", query])
        
        # Rewrite query
        user_input_json = json.dumps({"Query": query})
        rewritten_query = self.rewrite_query(user_input_json)
        
        # Get similar documents with metadata
        similar_docs = self.vectorstore.similarity_search_by_vector(
            self.embedding_model.embed_query(rewritten_query), 
            k=self.top_k
        )
        
        # Organize context with folder structure
        context_by_folder = {}
        for doc in similar_docs:
            folder = doc.metadata.get('folder_path', 'unknown')
            if folder not in context_by_folder:
                context_by_folder[folder] = []
            context_by_folder[folder].append(doc)
        
        # Build structured context
        structured_context = []
        for folder, docs in context_by_folder.items():
            folder_content = f"\nIn folder '{folder}':\n"
            for doc in docs:
                folder_content += f"- File '{doc.metadata['file_name']}':\n{doc.page_content}\n"
            structured_context.append(folder_content)
        
        retrieved_text = "\n".join(structured_context)
        
        # Enhanced prompt with better context understanding
        prompt = f"""You are an AI Note assistant. Use the following retrieved information to answer the question.
When answering questions about file locations or content organization:
- Mention the specific folders and file paths when relevant
- Provide a clear structure of where information is located
- If asked about folder contents, summarize what's in each relevant folder
- If asked about specific topics, mention which files contain that information

Retrieved Documents (Organized by Folder):
{retrieved_text}

Conversation History:
{self._format_conversation_history()}

Question: {rewritten_query}

Please provide a clear and structured answer that helps the user understand both the content and its organization in their notes.
"""

        if self.model_type == "ollama":
            response = self.generator.invoke(prompt)
        else:
            response = self.generator(prompt)
        
        self.conversation_history.append(["Assistant", response])
        return response

    def _format_conversation_history(self) -> str:
        """
        Format the conversation history for inclusion in the prompt.

        :param conversation_history: List of [sender, message] pairs.
        :return: Formatted conversation history string.
        """
        return "\n".join([f"{sender}: {message}" for sender, message in self.conversation_history])

    def add_documents(self, new_folder_path: str):
        """
        Add new documents from a different folder to the vector store.

        :param new_folder_path: Path to the new folder containing documents.
        """
        new_documents = load_documents(new_folder_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        new_splits = text_splitter.split_documents(new_documents)
        self.vectorstore.add_documents(new_splits)
        print(f"Added {len(new_splits)} new document chunks to the vector store.")

if __name__ == "__main__":
    # Initialize RAG with Ollama model and notes directory
    rag = RAGModel(model_type="Ollama", note_folder_path="../test/langchain_test")

    while True:
        user_input = input("You: ")
        response = rag.answer_query(user_input)
        print("Assistant:", response.content)