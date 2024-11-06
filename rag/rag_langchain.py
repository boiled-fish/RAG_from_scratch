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

from utils.custom_logger import CustomLogger
from utils.rag_utils import load_documents

class RAGModel:
    def __init__(self, model_type: str, note_folder_path: str, top_k: int = 5):
        """
        Initialize the RAG system.

        :param model_type: Type of the generator model ('ChatGPT-4' or 'Ollama').
        :param note_folder_path: Path to the folder containing notes/documents.
        :param top_k: Number of top relevant documents to retrieve.
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
        self.top_k = top_k

        # Initialize conversation history
        self.conversation_history = []

        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # Load and process documents
        documents = load_documents(self.note_folder_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.all_splits = text_splitter.split_documents(documents)

        # Initialize vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.all_splits,
            embedding=self.embedding_model,
            persist_directory=None  # In-memory; set path to persist
        )

        # Initialize generator model
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
        Answer the user's query based on the conversation history and retrieved documents.

        :param query: The user's current query.
        :param conversation_history: List of [sender, message] pairs.
        :return: The generated response.
        """
        if attached_files:
            query = f"\n\nOriginal Query: {query}\n\nAttached Files:\n\n"
            for file in attached_files:
                content = "\n".join(file['file_content'])
                query += f"{file['file_name']}:\n{content}\n\n"
        
        # Update conversation history with the new query
        self.log.info(f"Updating conversation history with query: {query}")
        self.conversation_history.append(["User", query])

        # Rewrite the query
        user_input_json = json.dumps({"Query": query})
        rewritten_query = self.rewrite_query(user_input_json)

        # Embed the rewritten query
        query_embedding = self.embedding_model.embed_query(rewritten_query)
        self.log.info(f"Embedded rewritten query.")

        # Retrieve top_k similar documents
        similar_docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.top_k)
        self.log.info(f"Retrieved {len(similar_docs)} similar documents.")

        # Combine retrieved documents into a single context
        retrieved_text = "\n\n".join([doc.page_content for doc in similar_docs])
        self.log.info(f"Combined retrieved documents into a single context.")

        # Construct the prompt
        prompt = f"""You are an AI Note assistant. Use the following retrieved information to answer the question.

Retrieved Documents:
{retrieved_text}

Conversation History:
{self._format_conversation_history()}

Question: {rewritten_query}

IMPORTANT: Answer ONLY the question, and nothing else. DO NOT include any other text or formatting.
"""

        # Generate the response
        if self.model_type == "ollama":
            response = self.generator.invoke(prompt)
        else:
            response = self.generator(prompt)

        # Update conversation history with the response
        self.conversation_history.append(["Assistant", response])
        self.log.info(f"Updated conversation history with response.")
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
        original_note_folder_path = self.note_folder_path
        self.note_folder_path = new_folder_path
        new_documents = load_documents(self.note_folder_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        new_splits = text_splitter.split_documents(new_documents)
        self.vectorstore.add_documents(new_splits)
        self.note_folder_path = original_note_folder_path
        print(f"Added {len(new_splits)} new document chunks to the vector store.")

if __name__ == "__main__":
    # Initialize RAG with Ollama model and notes directory
    rag = RAGModel(model_type="Ollama", note_folder_path="../test/langchain_test")

    while True:
        user_input = input("You: ")
        response = rag.answer_query(user_input)
        print("Assistant:", response.content)