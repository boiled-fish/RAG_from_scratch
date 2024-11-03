import warnings
import os
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QTextEdit, QComboBox, 
    QLabel, QFrame, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QMimeData, QEvent
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QTextDocument

from langchain.document_loaders import TextLoader, UnstructuredFileLoader

from utils.custom_logger import CustomLogger
from rag.rag_langchain import RAGModel
from utils.rag_utils import PDFLoader

class ChatMessage(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(self)

        message = QLabel(text)
        message.setWordWrap(True)
        message.setStyleSheet("""
            padding: 10px;
            border-radius: 15px;
            background-color: #FFFFFF;
            color: black;
        """ if is_user else """
            padding: 10px;
            border-radius: 15px;
            background-color: #FFFFFF;
            color: black;
        """)

        if is_user:
            layout.addStretch()
        layout.addWidget(message)
        if not is_user:
            layout.addStretch()

class NoteHelper(QMainWindow):
    def __init__(self, model_type: str, note_folder_path: str):
        super().__init__()

        self.logger = CustomLogger(log_dir_base='./log', logger_name="note_helper")
        self.log_dir = self.logger.get_log_dir()
        self.log = self.logger.get_logger()

        self.setWindowTitle("AI Chat Assistant")
        self.setGeometry(100, 100, 1000, 700)

        # 设置窗口背景颜色和其他组件的样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1C1C1C;
            }
            QComboBox {
                border: 2px solid #1C1C1C;
                border-radius: 0px;
                padding: 5px;
                background-color: white;
                min-width: 150px;
                min-height: 30px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 15px;
                color: black;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                color: black;
            }
            QComboBox::QAbstractItemView {
                color: white;
                background-color: #1C1C1C;
            }
            QTextEdit {
                border: 2px solid #E9ECEF;
                border-radius: 8px;
                padding: 5px;
                background-color: black;
                font-size: 14px;
                color: white;
            }
            QPushButton {
                background-color: #1C1C1C;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 15px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #0056b3;
                color: black;
            }
            QPushButton:pressed {
                background-color: #004494;
                color: black;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)

        self.setup_ui()

        self.model_type = model_type
        self.note_folder_path = note_folder_path
        self.log.info(f"Initializing NoteHelper with model_type: {model_type} and note_folder_path: {note_folder_path}")
        self.model = RAGModel(model_type=model_type, note_folder_path=note_folder_path)

        self.current_files = []
        self.current_files_num = 0

    def setup_ui(self):
        # 头部，选择模型
        header = QWidget()
        header.setStyleSheet("background-color: #1C1C1C; border-radius: 10px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 15, 15, 15)

        model_label = QLabel()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Ollama", "ChatGPT-4"])

        header_layout.addWidget(model_label)
        header_layout.addWidget(self.model_combo)
        header_layout.addStretch()

        self.layout.addWidget(header)

        # 聊天显示区域，带滚动条
        chat_container = QWidget()
        chat_container.setStyleSheet("""
            QWidget {
                background-color: #1C1C1C;
                border-radius: 10px;
            }
        """)
        chat_layout = QVBoxLayout(chat_container)

        self.chat_area = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_area)
        self.chat_layout.addStretch()

        self.scroll = QScrollArea()  # 将 QScrollArea 存储为实例变量
        self.scroll.setWidget(self.chat_area)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #F8F9FA;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #CED4DA;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

        chat_layout.addWidget(self.scroll)  # 使用实例变量
        self.layout.addWidget(chat_container)

        # 输入区域
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: #535953;
                border-radius: 10px;
                border: 0px solid #E9ECEF;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(15, 10, 15, 10)  # 调整上下边距以适应更小的高度

        # 用户输入控件
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("输入您的消息或拖放文件到此处...")
        self.user_input.setAcceptDrops(True)
        self.user_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.user_input.setMinimumHeight(30)  # 设置最小高度，使其至少一行高
        self.user_input.textChanged.connect(self.adjust_input_height)
        self.user_input.dragEnterEvent = self.dragEnterEvent
        self.user_input.dropEvent = self.dropEvent

        # 发送按钮
        self.send_button = QPushButton("↑")
        self.send_button.setStyleSheet("background-color: #1C1C1C; \
                                       color: white; \
                                       border: none; \
                                       border-radius: 8px; \
                                       padding: 5px 15px; \
                                       font-weight: bold; \
                                       min-height: 30px;")
        self.send_button.setFixedWidth(80)
        self.send_button.setFixedHeight(30)  # 与 QTextEdit 最小高度一致
        self.send_button.clicked.connect(self.send_message)

        # 将用户输入控件和发送按钮添加到输入布局
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)

        self.layout.addWidget(input_container)
        self.user_input.installEventFilter(self)

    def adjust_input_height(self):
        doc = self.user_input.document()
        doc.setTextWidth(self.user_input.viewport().width())
        height = doc.size().height()
        margin = self.user_input.contentsMargins().top() + self.user_input.contentsMargins().bottom()
        new_height = height + margin + 10  # 添加一些额外的空间

        # 设置最小和最大高度
        min_height = 30  # 单行高度
        max_height = 200  # 最大高度限制

        if new_height < min_height:
            new_height = min_height
        elif new_height > max_height:
            new_height = max_height

        self.user_input.setFixedHeight(new_height)

    def add_message(self, text, is_user=True):
        message = ChatMessage(text, is_user)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message)
        # 自动滚动到底部
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.pdf', '.txt', '.md', '.docx', '.pptx', '.xlsx', '.xls', 'py', 'ipynb', 'csv', 'json')):
                self.process_file(file_path)
            else:
                self.add_message(f"[WARNING] Unsupported file type: {file_path}", False)

    def process_file(self, file_path):
        self.log.info(f"Processing file: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        file_data = {
            "file_name": file_name,
            "file_content": []
        }
        try:
            if ext == '.txt' or ext == '.md':
                loader = TextLoader(file_path, encoding='utf-8')
                loader_docs = loader.load()
                file_data["file_content"].extend([doc.page_content for doc in loader_docs])
            elif ext == '.pdf':
                file_data["file_content"].extend(PDFLoader(file_path))
            else:
                loader = UnstructuredFileLoader(file_path)
                loaded_docs = loader.load()
                file_data["file_content"].extend([doc.page_content for doc in loaded_docs])
        except Exception as e:
            self.log.error(f"[ERROR] Error processing file: {file_path}: {e}")  

        self.add_message(f"[INFO] Attached file: {file_name}", False)
        self.current_files.append(file_data)
        self.current_files_num += 1

    def send_message(self):
        user_message = self.user_input.toPlainText().strip()
        
        if user_message:
            self.add_message(user_message, True)
            self.user_input.clear()

            selected_model = self.model_combo.currentText()

            bot_response = self.get_chatbot_response(user_message, self.current_files, selected_model)
            self.current_files = []
            self.current_files_num = 0

            self.add_message(bot_response, False)

    def get_chatbot_response(self, user_input, attached_files, model):
        if model != self.model_type:
            self.model_type = model
            self.model.switch_model(model)

        return self.model.answer_query(user_input, attached_files).content

    def eventFilter(self, obj, event):
        if obj == self.user_input and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+Enter: 允许插入新行
                    return False
                else:
                    # Enter: 发送消息
                    self.send_message()
                    return True  # 事件已处理
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("微软雅黑", 10)
    app.setFont(font)

    chatbot = NoteHelper(model_type="Ollama", note_folder_path="./test/langchain_test")
    chatbot.show()
    sys.exit(app.exec_())
