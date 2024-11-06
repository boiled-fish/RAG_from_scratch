from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_langchain import RAGModel
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: str = "ollama"
    files: list[dict] = []

class FolderRequest(BaseModel):
    folder_path: str

class NotesWatcher(FileSystemEventHandler):
    def __init__(self, rag_model):
        self.rag_model = rag_model
        self.last_processed = set()  # Track processed files
        
    def on_created(self, event):
        if not event.is_directory:
            # 新文件被创建
            file_path = Path(event.src_path)
            parent_dir = file_path.parent
            if file_path not in self.last_processed:
                print(f"New file detected: {file_path}")
                self.rag_model.add_documents(str(parent_dir))
                self.last_processed.add(file_path)

    def on_modified(self, event):
        if not event.is_directory:
            # 文件被修改
            file_path = Path(event.src_path)
            parent_dir = file_path.parent
            print(f"File modified: {file_path}")
            self.rag_model.add_documents(str(parent_dir))

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
            # 对于用户选择的文件夹，构建相对于当前工作目录的路径
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
async def chat(request: ChatRequest):
    if rag_model is None:
        raise HTTPException(
            status_code=400, 
            detail="RAG model not initialized. Please select a notes folder first."
        )
    
    try:
        # Switch model if needed
        if request.model.lower() != rag_model.model_type:
            rag_model.switch_model(request.model.lower())
        
        # Generate response
        response = rag_model.answer_query(
            query=request.message,
            attached_files=request.files
        )
        
        return {"response": response.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 