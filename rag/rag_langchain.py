import os
import json
from typing import List, Dict, Optional
import PyPDF2
import re
import logging
import sys
from dataclasses import dataclass
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

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

            # Create file handler with UTF-8 encoding
            fh = logging.FileHandler(f"{self.log_dir}/{logger_name}.log", encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Create console handler with UTF-8 encoding
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            # Ensure console output uses UTF-8
            sys.stdout.reconfigure(encoding='utf-8')
            self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger

    def get_log_dir(self):
        return self.log_dir

def PDFLoader(file_path: str):
    try:
        # Use langchain's PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Extract and clean text
        text_chunks = []
        for page in pages:
            text = page.page_content
            
            if not text:
                continue
                
            # Basic cleanup
            text = text.strip()
            if not text:
                continue
                
            # Remove control characters, but keep basic punctuation and line breaks
            text = ''.join(char for char in text if char >= ' ' or char in '\n\t')
            
            # Normalize whitespace characters
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Check if text is valid
            if len(text) < 10:  # Ignore short text
                continue
                
            # Check if text is all special characters
            valid_chars = sum(1 for c in text if c.isalnum() or c.isspace())
            if valid_chars / len(text) < 0.3:  # If valid characters are less than 30%, it's likely gibberish
                continue
            
            # Process in paragraphs
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if len(para) >= 10:  # Ensure paragraph has enough length
                    text_chunks.append(para)
        
        # Final check for all text chunks
        valid_chunks = []
        for chunk in text_chunks:
            # Clean and validate again
            chunk = chunk.strip()
            if len(chunk) >= 10 and any(c.isalnum() for c in chunk):
                valid_chunks.append(chunk)
        
        return valid_chunks
        
    except Exception as e:
        print(f"[ERROR] Error processing PDF file {file_path}: {str(e)}")
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
                        "folder_name": os.path.basename(root),
                        "folder_structure": folder_structure
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

@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()

@dataclass
class FileInfo:
    path: str
    name: str
    type: str
    folder: str
    content_chunks: List[Document]

class FileStructureManager:
    def __init__(self):
        self.files: Dict[str, FileInfo] = {}
        self.folder_structure: Dict[str, List[str]] = {}
    
    def add_file(self, file_path: str, file_name: str, file_type: str, 
                 folder: str, content_chunks: List[Document]):
        """Add a file to the structure"""
        self.files[file_name.lower()] = FileInfo(
            path=file_path,
            name=file_name,
            type=file_type,
            folder=folder,
            content_chunks=content_chunks
        )
        
        # Update folder structure
        if folder not in self.folder_structure:
            self.folder_structure[folder] = []
        self.folder_structure[folder].append(file_name)
    
    def get_file_info(self, file_name: str) -> Optional[FileInfo]:
        """Get file info by name (case insensitive)"""
        return self.files.get(file_name.lower())
    
    def search_file(self, partial_name: str) -> List[FileInfo]:
        """Search files by partial name"""
        partial_name = partial_name.lower()
        return [
            file_info for file_name, file_info in self.files.items()
            if partial_name in file_name
        ]

@dataclass
class ModelStatus:
    is_processing: bool = False
    current_operation: str = ""

class RAGModel:
    def __init__(self, model_type: str, note_folder_path: str, top_k: int = 20):
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
        self.top_k = top_k

        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # Initialize file structure manager
        self.file_manager = FileStructureManager()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        # Load and process documents with file structure tracking
        self.log.info(f"Loading documents from {note_folder_path}")
        documents = load_documents(self.note_folder_path)
        
        # Group documents by file
        file_documents: Dict[str, List[Document]] = {}
        for doc in documents:
            file_name = doc.metadata['file_name']
            if file_name not in file_documents:
                file_documents[file_name] = []
            file_documents[file_name].append(doc)
        
        # Add files to manager and process chunks
        all_splits = []
        for file_name, docs in file_documents.items():
            # Split documents
            splits = self.text_splitter.split_documents(docs)
            all_splits.extend(splits)
            
            # Add to file manager
            if splits:
                self.file_manager.add_file(
                    file_path=splits[0].metadata['source'],
                    file_name=file_name,
                    file_type=splits[0].metadata['file_type'],
                    folder=splits[0].metadata['folder_name'],
                    content_chunks=splits
                )

        # Use split_documents to preserve metadata
        splits = self.text_splitter.split_documents(all_splits)
        self.log.info(f"Created {len(splits)} text chunks")

        # Initialize vector store with FAISS
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embedding_model
        )
        self.log.info("Initialized FAISS vector store")

        # Initialize the appropriate model
        self._initialize_generator_model()
        self.log.info(f"Initialized {self.model_type} model")

        # Initialize chat history
        self.chat_history: List[Message] = []
        self.max_history = 10  # Keep last 10 messages for context

        self.status = ModelStatus()

    def _initialize_generator_model(self):
        """
        Initialize the generator model based on the current model_type.
        """
        if self.model_type == "chatgpt-4":
            self.generator = ChatOpenAI(model_name="gpt-4", temperature=0.7)
            self.log.info("Initialized ChatGPT-4 model.")
        elif self.model_type == "ollama":
            self.generator = ChatOllama(model="llama3.1:8b", temperature=0.5)
            self.log.info("Initialized Ollama model.")
        else:
            self.log.error("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")
            raise ValueError("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")
        

    def switch_model(self, new_model_type: str):
        """
        Switch the generator model to a different type.

        :param new_model_type: The new model type ('ChatGPT-4').
        """
        if new_model_type.lower() not in ["chatgpt-4", "ollama"]:
            self.log.error("Unsupported model_type. Choose 'ChatGPT-4' or 'Ollama'.")
            return False
        
        self.model_type = new_model_type.lower()
        self._initialize_generator_model()
        self.log.info(f"Switched to model: {self.model_type}")

        return True

    def _get_conversation_context(self) -> str:
        """Format recent conversation history into a string"""
        if not self.chat_history:
            return ""
            
        formatted_history = []
        for msg in self.chat_history[-self.max_history:]:
            role = "Human" if msg.role == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg}")
        
        return "\n".join(formatted_history)

    def answer_query(self, query: str, attached_files: List[dict] = []) -> str:
        """
        Enhanced answer generation with better context understanding.
        """
        try:
            if attached_files:
                query = f"\n\nOriginal Query: {query}\n\nAttached Files:\n\n"
                for file in attached_files:
                    if file.get("file_path"):
                        chunks = PDFLoader(file["file_path"])
                        if chunks:
                            query += f"{file['file_name']}:\n{chunks}\n\n"

            self.log.info(f"Query: {query}")
            
            # Add user message to history
            self.chat_history.append(Message(role="user", content=query))
            
            # First, check if query is about specific files
            file_check_prompt = """Analyze if this query is asking about specific files.
            If it is, extract the filename(s). If not, return "NO_FILE".
            
            Query: {query}
            
            Return format:
            If about specific files: FILENAME: <extracted_filename>
            If not about specific files: NO_FILE"""
            
            file_check_response = self.generator.invoke(file_check_prompt.format(query=query))
            file_check_result = str(file_check_response).strip()
            
            similar_docs = []
            if file_check_result.startswith("FILENAME:"):
                # Extract filename and get file-specific documents
                target_filename = file_check_result.split("FILENAME:")[1].strip()
                self.log.info(f"Query is about specific file: {target_filename}")
                
                # Search for the file
                matching_files = self.file_manager.search_file(target_filename)
                if matching_files:
                    # Get all content chunks from matching files
                    for file_info in matching_files:
                        similar_docs.extend(file_info.content_chunks)
                    
                    # If we have too many chunks, prioritize most relevant ones
                    if len(similar_docs) > self.top_k:
                        # Get embeddings for query
                        query_embedding = self.vectorstore.embedding_function(query)
                        
                        # Sort chunks by relevance
                        chunk_scores = []
                        for doc in similar_docs:
                            doc_embedding = self.vectorstore.embedding_function(doc.page_content)
                            score = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                            chunk_scores.append((score, doc))
                        
                        # Get top_k most relevant chunks
                        similar_docs = [doc for _, doc in sorted(chunk_scores, reverse=True)[:self.top_k]]
            
            # If no file-specific docs or no file mentioned, use regular similarity search
            if not similar_docs:
                similar_docs = self.vectorstore.similarity_search(query, k=self.top_k)
            
            # Build context from similar documents
            context_parts = []
            for doc in similar_docs:
                metadata = doc.metadata
                folder_path = "/".join(metadata.get('folder_structure', []))
                context_parts.append(
                    f"File: {metadata['file_name']} (in folder: {metadata['folder_name']})\n"
                    f"Content: {doc.page_content}\n"
                )
            
            context = "\n".join(context_parts)
            self.log.info(f"Context: {context}")

            # Create prompt with conversation history and document context
            conversation_context = self._get_conversation_context()
            prompt = f"""You are an assistant that provides informative answers using the given context.

If the context lacks information relevant to the question, disregard irrelevant details.

Conversation history [text between "" are conversation history]:
"{conversation_context}"

Context [text between "" are context]:
"{context}"

Question [text between "" are question]:
"{query}"

Please provide a clear and helpful answer.
- Consider the conversation history in your response.
- Do not mention the content of the prompt or refer explicitly to the context.
- Avoid repeating information.
- Do not state that you've been asked similar questions before or that you have access to certain documents.
            """

            # Get response from model
            response = self.generator.invoke(prompt)
            #response_content = str(response.content if hasattr(response, 'content') else response)

            # Add assistant response to history
            self.chat_history.append(Message(role="assistant", content=response))
            
            return response
            
        except Exception as e:
            self.log.error(f"Error in answer_query: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def add_documents(self, new_folder_path: str):
        """
        Add new documents from a different folder to the vector store.
        """
        try:
            self.status.is_processing = True
            self.status.current_operation = "Processing new documents..."
            self.log.info("Started processing new documents")
            
            # Load new documents
            new_documents = load_documents(new_folder_path)
            if not new_documents:
                self.log.warning(f"No new documents found in {new_folder_path}")
                return

            # Split documents
            self.status.current_operation = "Splitting documents..."
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            new_splits = text_splitter.split_documents(new_documents)
            
            if not new_splits:
                self.log.warning("No text chunks created after splitting")
                return

            # Create new vectorstore
            self.status.current_operation = "Updating vector store..."
            new_vectorstore = FAISS.from_documents(
                documents=new_splits,
                embedding=self.embedding_model
            )
            
            # Merge vectorstores
            self.vectorstore.merge_from(new_vectorstore)
            
            self.log.info(f"Added {len(new_splits)} new document chunks to the vector store")
            
        except Exception as e:
            self.log.error(f"Error adding documents: {str(e)}")
            raise e
        finally:
            self.status.is_processing = False
            self.status.current_operation = ""

    def clear_history(self):
        """Clear conversation history"""
        self.chat_history = []

if __name__ == "__main__":
    # Initialize RAG with ChatGPT-4 model and notes directory
    rag = RAGModel(model_type="ChatGPT-4", note_folder_path="../test/langchain_test")

    while True:
        user_input = input("You: ")
        response = rag.answer_query(user_input)
        print("Assistant:", response)
