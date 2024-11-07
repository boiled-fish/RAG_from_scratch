# Note Helper

Note Helper is an AI-powered web application that helps you manage and interact with your notes using natural language processing. Built with Next.js and FastAPI, it provides an intuitive interface for document analysis and question answering.

## Features

- 🤖 AI-powered note analysis and question answering
- 📁 Support for multiple document formats (PDF, TXT, MD)
- 🔄 Real-time file processing status updates
- 📎 Drag-and-drop file attachments
- 💬 Interactive chat interface
- 🔄 Multiple AI model support (Ollama, ChatGPT-4)

## Tech Stack

- Frontend: Next.js 15.0
- UI Components: shadcn/ui
- Styling: Tailwind CSS
- Backend: FastAPI
- AI Models: Ollama, ChatGPT-4

## Environment Setup

1. Make sure you have Node.js installed (v18 or higher)
2. Install Python dependencies with conda:
```bash
conda create -n note-helper python=3.9.19
conda activate note-helper
pip install -r server/requirements.txt
```

3. Install Ollama, see [Ollama](https://ollama.com/docs/installation) && Pull Ollama models [Ollama3.1:8b, Nomic-embed-text-v3.small]
```bash
ollama pull ollama/llama3.1:8b
ollama pull ollama/nomic-embed-text-v3.small
```

4. Set up the required Ollama or OpenAI API key in the .env file
```bash
OLLAMA_HOST=http://localhost:11434 # Default ollama server host, change if you have a remote server
OPENAI_API_KEY= # OpenAI API key, change if you want to use OpenAI
```

## Getting Started

1. Clone the repository & cd note-helper-rag
2. Launch Ollama server, see [Ollama](https://ollama.com/docs/installation) and make sure Ollama models [Ollama3.1:8b, Nomic-embed-text-v3.small] are pulled
```bash
ollama pull ollama/llama3.1:8b
ollama pull ollama/nomic-embed-text-v3.small
```

3. Install React dependencies:
```bash
npm install
```

4. Start the development server:
```bash
npm run dev
```

5. Start the backend server:
```bash
uvicorn server:app --reload --port 8000
```

6. Open [http://localhost:3000](http://localhost:3000) with your browser

## Project Structure

```
note-helper-rag/
├── src/
│   ├── app/
│   │   ├── global.css    # Global styles
│   │   ├── layout.tsx    # App layout and metadata
│   │   └── page.tsx      # Main page component
│   │   └── favicon.ico   # Favicon
│   └── components/
│   |   ├── note_helper.tsx    # Main application component
│   |   └── ui/               # UI components
│   └── server/                   # Backend server code
│       ├── server.py               # FastAPI application
│       ├── rag_langchain.py        # Langchain application
│       └── requirements.txt        # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.