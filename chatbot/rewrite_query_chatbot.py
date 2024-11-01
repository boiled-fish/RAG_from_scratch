import sys
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Get the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

import json
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from langchain.schema import Document
import PyPDF2
from rag.rag_langchain import RAGModel



# PyQt5 application class
class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chatbot")
        self.setAcceptDrops(True)  # Enable drag and drop

        # Initialize conversation history as a list of tuples
        self.conversation_history = []  # List[Tuple[str, str]]

        # Initialize RAGModel with default parameters
        self.rag_model = RAGModel(k=5, generator_model='llama3:latest')

        # Build the UI
        self.init_ui()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(['llama3:latest', 'gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini'])
        self.model_combo.currentIndexChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        # Conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        main_layout.addWidget(self.conversation_display)

        # Input layout
        input_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your message here...")
        self.input_line.returnPressed.connect(self.send_message)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(send_button)
        main_layout.addLayout(input_layout)

        # Set window size
        self.resize(600, 800)

    def change_model(self):
        selected_model = self.model_combo.currentText()
        self.rag_model.generator_model = selected_model
        self.rag_model.generator = self.rag_model._get_generator()
        self.append_to_conversation("System", f"Generator model changed to {selected_model}")

    def send_message(self):
        user_input = self.input_line.text().strip()
        if user_input:
            self.input_line.clear()
            self.process_user_input(user_input)
        else:
            pass  # Do nothing if input is empty

    def process_user_input(self, user_input):
        # Prepare the conversation history
        if len(self.conversation_history) > 1:
            # Create user_input_json
            user_input_json = json.dumps({"Query": user_input})
            # Call rewrite_query
            rewritten_query_json = self.rewrite_query(user_input_json, self.conversation_history)
            rewritten_query_data = json.loads(rewritten_query_json)
            rewritten_query = rewritten_query_data["Rewritten Query"]
            # Log the rewritten query
            self.append_to_conversation("System", f"Rewritten Query: {rewritten_query}")
        else:
            rewritten_query = user_input

        # Append user message to the conversation display
        self.append_to_conversation("User", user_input)
        # Update conversation history
        self.conversation_history.append(("User", user_input))

        # Prepare chat history for generate method (excluding the last user input)
        chat_history = self.conversation_history[:-1]

        # Generate response using RAGModel
        response = self.rag_model.generate(rewritten_query, chat_history=chat_history)
        assistant_response = response['answer']

        # Append assistant's response to the conversation display
        self.append_to_conversation("Assistant", assistant_response)
        # Update conversation history
        self.conversation_history.append(("Assistant", assistant_response))

    def append_to_conversation(self, speaker, message):
        self.conversation_display.append(f"<b>{speaker}:</b> {message}")

    # The rewrite_query function as specified
    def rewrite_query(self, user_input_json, conversation_history):
        user_input = json.loads(user_input_json)["Query"]
        # Get the last two messages from conversation history
        context_messages = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
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
        response = self.rag_model.generator(prompt)
        rewritten_query = response.strip()
        return json.dumps({"Rewritten Query": rewritten_query})

    # Drag and drop events
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        # Handle file drop
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file_path in files:
            self.process_file(file_path)
        event.acceptProposedAction()

    def process_file(self, file_path):
        # Determine file type and process accordingly
        if file_path.lower().endswith('.pdf'):
            self.convert_pdf_to_text(file_path)
        elif file_path.lower().endswith('.txt'):
            self.upload_txtfile(file_path)
        elif file_path.lower().endswith('.json'):
            self.upload_jsonfile(file_path)
        else:
            self.append_to_conversation("System", f"Unsupported file type: {file_path}")

    def convert_pdf_to_text(self, file_path):
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
                # Add chunks to RAGModel
                documents = [Document(page_content=chunk.strip()) for chunk in chunks]
                self.rag_model.add_documents(documents)
                self.append_to_conversation("System", f"PDF content from {os.path.basename(file_path)} has been added to the knowledge base.")
        except Exception as e:
            self.append_to_conversation("System", f"Error processing PDF file: {e}")

    def upload_txtfile(self, file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as txt_file:
                text = txt_file.read()
                
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
                # Add chunks to RAGModel
                documents = [Document(page_content=chunk.strip()) for chunk in chunks]
                self.rag_model.add_documents(documents)
                self.append_to_conversation("System", f"Text file content from {os.path.basename(file_path)} has been added to the knowledge base.")
        except Exception as e:
            self.append_to_conversation("System", f"Error processing text file: {e}")

    def upload_jsonfile(self, file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                
                # Flatten the JSON data into a single string
                text = json.dumps(data, ensure_ascii=False)
                
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
                # Add chunks to RAGModel
                documents = [Document(page_content=chunk.strip()) for chunk in chunks]
                self.rag_model.add_documents(documents)
                self.append_to_conversation("System", f"JSON file content from {os.path.basename(file_path)} has been added to the knowledge base.")
        except Exception as e:
            self.append_to_conversation("System", f"Error processing JSON file: {e}")

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    window = ChatbotWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
