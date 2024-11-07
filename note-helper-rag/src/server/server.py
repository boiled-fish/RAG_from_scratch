from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_langchain import RAGModel
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import threading
from typing import List
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    model: str = "ollama"
    files: list[dict] = []

class FolderRequest(BaseModel):
    folder_path: str

class NotesWatcher(FileSystemEventHandler):
    def __init__(self, rag_model):
        self.rag_model = rag_model
        self.last_processed = {}  # 使用字典存储文件最后处理时间
        self.processing_lock = threading.Lock()
        self.cooldown_period = 5  # 冷却时间（秒）
    
    def should_process_file(self, file_path: Path) -> bool:
        """检查文件是否应该被处理"""
        current_time = time.time()
        
        # 检查文件扩展名
        if file_path.suffix.lower() not in ['.txt', '.md', '.pdf']:
            return False
            
        # 检查是否在冷却期内
        if file_path in self.last_processed:
            last_time = self.last_processed[file_path]
            if current_time - last_time < self.cooldown_period:
                return False
        
        return True
    
    def process_file_change(self, file_path: Path):
        """处理文件变更"""
        with self.processing_lock:
            if not self.should_process_file(file_path):
                return
                
            try:
                print(f"Processing file: {file_path}")
                parent_dir = file_path.parent
                self.rag_model.add_documents(str(parent_dir))
                self.last_processed[file_path] = time.time()
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        self.process_file_change(file_path)

    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        self.process_file_change(file_path)

# Global variables
rag_model = None
observer = None

@app.post("/init_notes")
async def init_notes(request: FolderRequest):
    global rag_model, observer
    try:
        # 获取当前工作目录的绝对路径
        current_dir = Path.cwd()
        folder_path = Path(request.folder_path.lstrip('/'))  # 移除前导斜杠
        
        # 如果是默认路径 ./notes
        if str(folder_path) == "notes":
            abs_folder_path = current_dir / "notes"
        else:
            # 对于用户选择文件夹，构建相对于当前工作目录的路径
            abs_folder_path = current_dir / folder_path
        
        # 确保目录存在
        abs_folder_path.mkdir(exist_ok=True)
        
        print(f"[INFO] Using absolute path: {abs_folder_path}")
        
        # Initialize RAGModel with absolute path
        rag_model = RAGModel(
            model_type="ollama",
            note_folder_path=str(abs_folder_path),
            top_k=5
        )
        
        # Set up the file watcher with absolute path
        if observer:
            observer.stop()
        event_handler = NotesWatcher(rag_model)
        observer = Observer()
        observer.schedule(event_handler, str(abs_folder_path), recursive=True)
        observer.start()
        
        return {
            "message": f"Notes directory initialized at {abs_folder_path}",
            "success": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize RAG model: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    global observer
    if observer:
        observer.stop()
        observer.join()

@app.post("/chat")
async def chat(
    message: str = Form(...),
    model: str = Form("ollama"),
    files: List[UploadFile] = File([])
):
    if rag_model is None:
        raise HTTPException(status_code=400, detail="RAG model not initialized")
    
    try:
        if model.lower() != rag_model.model_type:
            rag_model.switch_model(model.lower())
        
        # 处理上传的文件
        processed_files = []
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                # 使用 os.path.join 构建文件路径
                file_path = os.path.join(TEMP_DIR, file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                processed_files.append({
                    "file_name": file.filename,
                    "file_path": file_path
                })
        
        response = rag_model.answer_query(
            query=message,
            attached_files=processed_files
        )
        
        return {"response": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def get_model_status():
    if rag_model is None:
        return {"is_processing": False, "operation": "Model not initialized"}
    return {
        "is_processing": rag_model.status.is_processing,
        "operation": rag_model.status.current_operation
    }