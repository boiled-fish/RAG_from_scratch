# Note Helper

<div align="center">

An AI-powered note management system that enables natural language interactions with your documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Next.js](https://img.shields.io/badge/Next.js-14.0-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688)](https://fastapi.tiangolo.com/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/downloads/)

</div>

## ✨ Features

- 🤖 **AI-Powered Analysis**: Advanced note analysis and intelligent question answering
- 📁 **Multi-Format Support**: Seamlessly handle PDF, TXT, and MD files
- 🔄 **Real-Time Processing**: Live status updates during file processing
- 📎 **Easy File Management**: Intuitive drag-and-drop file uploads
- 💬 **Interactive Chat**: Natural conversation interface with your documents
- 🔄 **Flexible AI Models**: Switch between Ollama and OpenAI models
- 📊 **Vector Database**: Efficient document embedding and retrieval
- 🔍 **RAG Implementation**: State-of-the-art retrieval augmented generation

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 14.0
- **UI Components**: shadcn/ui
- **Styling**: Tailwind CSS

### Backend
- **Server**: FastAPI
- **AI Models**: 
  - Ollama (llama3.1:8b)
  - OpenAI (GPT-4)
- **Vector Store**: FAISS

## 📋 Prerequisites

- Node.js (v18 or higher)
- Python 3.9+
- Ollama (for local AI model support)
- Git

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/boiled-fish/RAG_from_scratch.git
cd RAG_from_scratch/note-helper-rag
```

### 2. Environment Setup

1. Create and activate Python environment:
```bash
conda create -n note-helper python=3.9.19
conda activate note-helper
pip install -r server/requirements.txt
```

2. Ollama Setup
2.1. Install Ollama from [official website](https://ollama.com/docs/installation)
2.2. Pull required models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
```

3. Environment Variables
Create a `.env` file in the root directory:
```bash
# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key
OLLAMA_BASE_URL=http://localhost:11434

# Langsmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```

4. Frontend Dependencies
```bash
npm install
```

### 3. Launch Application
1. Launch Ollama server
```bash
ollama serve
```

2. Start development server (in a new terminal):
```bash
npm run dev
```

3. Start backend server (in another terminal):
```bash
uvicorn server:app --reload --port 8000
```

4. Visit [http://localhost:3000](http://localhost:3000) to access the application.

## 📁 Project Structure

```
note-helper-rag/
├── src/                      # Source code
│   ├── app/                  # Next.js app
│   │   ├── globals.css       # Global styles
│   │   ├── layout.tsx        # App layout
│   │   └── page.tsx          # Main page
│   ├── components/           # React components
│   │   ├── note_helper.tsx   # Main app component
│   │   └── ui/               # UI components
│   └── server/               # Backend
│       ├── server.py         # FastAPI server
│       ├── rag_langchain.py  # RAG implementation
│       └── requirements.txt  # Python dependencies
├── py_client/                # Python desktop client
│   ├── note_helper.py        # PyQt5 client
└── package.json              # Node.js dependencies
```

## 💡 Usage Guide

### Note Management
1. **Select Note Path**: 
   - Use default path (`./note-helper-rag/note`)
   - Or choose custom directory
2. **Process Notes**:
   - System will automatically process the note file and save the embeddings to the vector database, and when you add new notes, system will also process the new notes and update the vector database.

### AI Model Selection
- Toggle between Ollama and OpenAI models via the dropdown in top-right corner
- Default: Ollama (llama3.1:8b)

### Interacting with Notes
- Ask questions about your documents
- System uses RAG to provide context-aware responses
- Supports general chat functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request