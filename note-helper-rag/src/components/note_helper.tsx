'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Upload, Send, Bot, User, Folder, Loader2, X, Paperclip } from 'lucide-react'

declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}

declare global {
  interface Window {
    showDirectoryPicker(): Promise<FileSystemDirectoryHandle>;
  }
}

declare global {
  interface FileSystemDirectoryHandle {
    entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
    requestPermission(descriptor: { mode: 'read' | 'readwrite' }): Promise<'granted' | 'denied'>
  }
}

interface Message {
  text: string
  isUser: boolean
  files?: Array<{
    file_name: string,
    file_content?: string[]
    file?: File
  }>
}

function Component() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState('Ollama')
  const [attachedFiles, setAttachedFiles] = useState<Array<{
    file_name: string,
    file: File
  }>>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showFolderDialog, setShowFolderDialog] = useState(true)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const [isModelInitialized, setIsModelInitialized] = useState(false)
  const [isProcessingNotes, setIsProcessingNotes] = useState(false)
  const [manualPath, setManualPath] = useState('')
  const [showManualInput, setShowManualInput] = useState(false)
  const [isProcessingFiles, setIsProcessingFiles] = useState(false);
  const [processingOperation, setProcessingOperation] = useState("");
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current;
      setTimeout(() => {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }, 50);
    }
  }, [messages, isLoading]);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/model_status');
        const data = await response.json();
        setIsProcessingFiles(data.is_processing);
        setProcessingOperation(data.operation);
      } catch (error) {
        console.error('Error polling status:', error);
      }
    };

    const intervalId = setInterval(pollStatus, 2000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current;
      requestAnimationFrame(() => {
        const scrollElement = scrollContainer.querySelector('[data-radix-scroll-area-viewport]');
        if (scrollElement) {
          scrollElement.scrollTop = scrollElement.scrollHeight;
        }
      });
    }
  }, [messages, isLoading, attachedFiles]);

  const handleSend = async () => {
    if (isProcessingNotes) {
      setMessages(prev => [...prev, { 
        text: "Please wait while I finish processing the notes...", 
        isUser: false 
      }]);
      return;
    }
    
    if (!isModelInitialized) {
      setMessages(prev => [...prev, { 
        text: "Please select a notes folder first", 
        isUser: false 
      }])
      setShowFolderDialog(true)
      return
    }
    
    if (input.trim() || attachedFiles.length > 0) {
      const userMessage = { text: input, isUser: true, files: attachedFiles }
      setMessages(prev => [...prev, userMessage])
      setInput('')
      setAttachedFiles([])
      setIsLoading(true)
  
      const formData = new FormData()
      formData.append('message', input)
      formData.append('model', model)
      
      // Modify file addition method
      attachedFiles.forEach((fileObj) => {
        formData.append('files', fileObj.file)  // Use 'files' as key, matching server side
      })

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          body: formData
        })
  
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Failed to get response from the server')
        }
  
        const data = await response.json()
        setMessages(prev => [...prev, { text: data.response, isUser: false }])
      } catch (error) {
        console.error('Error:', error)
        const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred'
        setMessages(prev => [...prev, { 
          text: `Error: ${errorMessage}`, 
          isUser: false 
        }])
      } finally {
        setIsLoading(false)
      }
    }
  }

  const handleFileAttach = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return
    
    const processedFiles = await Promise.all(Array.from(e.target.files).map(async file => {
      return {
        file_name: file.name,
        file: file
      }
    }))
    setAttachedFiles([...attachedFiles, ...processedFiles])
  }

  const handleUseDefault = async () => {
    try {
      setShowFolderDialog(false);
      setIsProcessingNotes(true);
      setMessages(prev => [...prev, { 
        text: "Assistant is processing notes...", 
        isUser: false 
      }]);
      
      const response = await fetch('http://localhost:8000/init_notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: './notes' })
      });
      
      if (response.ok) {
        setIsModelInitialized(true);
        setMessages(prev => [...prev, { 
          text: "Notes processing complete. How can I help you?", 
          isUser: false 
        }]);
      }
    } catch (error) {
      console.error('Error initializing default folder:', error);
      setMessages(prev => [...prev, { 
        text: "Error processing notes. Please try again.", 
        isUser: false 
      }]);
      setShowFolderDialog(true);
    } finally {
      setIsProcessingNotes(false);
    }
  }

  const handleManualPathSubmit = async () => {
    try {
      setShowFolderDialog(false);
      setIsProcessingNotes(true);
      setMessages(prev => [...prev, { 
        text: "Assistant is processing notes...", 
        isUser: false 
      }]);
      
      const response = await fetch('http://localhost:8000/init_notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: manualPath })
      });
      
      if (response.ok) {
        setIsModelInitialized(true);
        setMessages(prev => [...prev, { 
          text: "Notes processing complete. How can I help you?", 
          isUser: false 
        }]);
      } else {
        const errorData = await response.json();
        setMessages(prev => [...prev, { 
          text: `Error processing notes: ${errorData.detail}`, 
          isUser: false 
        }]);
        setShowFolderDialog(true);
      }
    } catch (error) {
      console.error('Error initializing with manual path:', error);
      setMessages(prev => [...prev, { 
        text: "Error processing notes. The folder might be too large or contain unsupported files.", 
        isUser: false 
      }]);
      setShowFolderDialog(true);
    } finally {
      setIsProcessingNotes(false);
      setShowManualInput(false);
      setManualPath('');
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    const processedFiles = await Promise.all(files.map(async file => ({
      file_name: file.name,
      file: file
    })))
    setAttachedFiles(prev => [...prev, ...processedFiles])
  }

  return (
    <>
      <Dialog 
        open={showFolderDialog} 
        onOpenChange={(open) => {
          if (!open && !isModelInitialized) {
            return;
          }
          setShowFolderDialog(open);
          setShowManualInput(false);
          setManualPath('');
        }}
      >
        <DialogContent className="bg-white max-w-md w-full p-6 rounded-xl border shadow-lg">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-gray-900">
              Select Notes Folder
            </DialogTitle>
            <p className="text-gray-600 mt-2">
              Please choose how you want to specify the notes folder:
            </p>
          </DialogHeader>
          
          <div className="flex flex-col gap-4 mt-6">
            {!showManualInput ? (
              <>
                <Button
                  variant="outline"
                  onClick={handleUseDefault}
                  className="w-full h-14 justify-start gap-3 text-gray-700 hover:bg-gray-50 border-gray-200"
                >
                  <Folder className="h-5 w-5 text-blue-500" />
                  Use Default
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowManualInput(true)}
                  className="w-full h-14 justify-start gap-3 text-gray-700 hover:bg-gray-50 border-gray-200"
                >
                  <Folder className="h-5 w-5 text-blue-500" />
                  Enter Path Manually
                </Button>
              </>
            ) : (
              <>
                <Input
                  type="text"
                  placeholder="e.g., C:\Users\YourName\Documents\Notes"
                  value={manualPath}
                  onChange={(e) => setManualPath(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleManualPathSubmit()}
                  className="border-gray-200"
                />
                <div className="flex gap-3">
                  <Button
                    onClick={handleManualPathSubmit}
                    className="flex-1 bg-blue-500 hover:bg-blue-600 text-white"
                    disabled={!manualPath}
                  >
                    Confirm
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setShowManualInput(false);
                      setManualPath('');
                    }}
                    className="flex-1 text-gray-700 border-gray-200 hover:bg-gray-50"
                  >
                    Back
                  </Button>
                </div>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <div 
        className="flex flex-col h-screen bg-gray-50"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
          <div className="absolute inset-0 bg-blue-500/20 backdrop-blur-sm z-50 flex items-center justify-center">
            <div className="bg-white p-6 rounded-xl shadow-lg text-center">
              <Upload className="h-12 w-12 text-blue-500 mx-auto mb-3" />
              <p className="text-lg font-medium text-gray-900">Drop files here</p>
            </div>
          </div>
        )}

        <Card className="m-4 bg-white border-gray-200 shadow-sm">
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <h1 className="text-2xl font-bold text-gray-900">Note Helper</h1>
              <Select 
                value={model} 
                onValueChange={setModel} 
                disabled={isProcessingNotes}
              >
                <SelectTrigger className="w-[180px] bg-white border-gray-200">
                  <SelectValue placeholder="Select Model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Ollama">Ollama</SelectItem>
                  <SelectItem value="ChatGPT-4">ChatGPT-4</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <ScrollArea 
          className="flex-grow mx-4 mb-4 p-4 rounded-xl bg-white border border-gray-200 shadow-sm overflow-y-auto" 
          style={{ maxHeight: 'calc(100vh - 200px)' }}
          ref={scrollAreaRef}
        >
          {messages.map((message, index) => (
            <div key={index} className={`mb-4 ${message.isUser ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block p-4 rounded-xl max-w-[80%] ${
                message.isUser 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 text-gray-900'
              }`}>
                <div className="flex items-center mb-3">
                  {message.isUser ? <User className="mr-2 h-5 w-5" /> : <Bot className="mr-2 h-5 w-5" />}
                  <span className="font-semibold">{message.isUser ? 'You' : 'AI'}</span>
                </div>
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {message.text.split('\n').map((paragraph, i) => {
                    // Process title (# starts)
                    if (paragraph.startsWith('# ')) {
                      return <h1 key={i} className="text-xl font-bold mb-2">{paragraph.substring(2)}</h1>;
                    }
                    // Process sub-title (## starts)
                    if (paragraph.startsWith('## ')) {
                      return <h2 key={i} className="text-lg font-bold mb-2">{paragraph.substring(3)}</h2>;
                    }
                    // Process list item (- or * starts)
                    if (paragraph.trim().startsWith('- ') || paragraph.trim().startsWith('* ')) {
                      return <li key={i} className="ml-4">{paragraph.substring(2)}</li>;
                    }
                    // Process bold text (**text**)
                    const boldText = paragraph.replace(
                      /\*\*(.*?)\*\*/g, 
                      '<strong>$1</strong>'
                    );
                    // Process code snippet (`code`)
                    const codeText = boldText.replace(
                      /`(.*?)`/g,
                      '<code class="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded">$1</code>'
                    );
                    
                    // If it's an empty line, add extra spacing
                    if (paragraph.trim() === '') {
                      return <div key={i} className="h-4"></div>;
                    }
                    
                    return (
                      <p 
                        key={i} 
                        className="mb-2 last:mb-0"
                        dangerouslySetInnerHTML={{ __html: codeText }}
                      />
                    );
                  })}
                </div>
                {message.files && message.files.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium mb-1">Attached files:</p>
                    <ul className="text-sm space-y-1">
                      {message.files.map((file, fileIndex) => (
                        <li key={fileIndex} className="flex items-center">
                          <span className="mr-2">📎</span>
                          {file.file_name}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="text-center text-gray-600">
              <p>Assistant is thinking...</p>
            </div>
          )}
        </ScrollArea>

        <div className="flex items-center gap-3 mx-4 mb-4">
          <Input
            type="text"
            placeholder={isProcessingNotes ? "Please wait..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            className="flex-grow bg-white border-gray-200"
            disabled={isProcessingNotes || isLoading}
          />
          <input
            type="file"
            id="file-upload"
            onChange={handleFileAttach}
            className="hidden"
            multiple
          />
          <Button
            variant="outline"
            size="icon"
            className="border-gray-200 hover:bg-gray-50"
            disabled={isProcessingNotes || isLoading}
          >
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="h-5 w-5 text-gray-500" />
            </label>
          </Button>
          <Button
            onClick={handleSend}
            disabled={isProcessingNotes || isLoading}
            className="bg-blue-500 hover:bg-blue-600 text-white px-6"
          >
            <Send className="mr-2 h-4 w-4" /> Send
          </Button>
        </div>
        {attachedFiles.length > 0 && !isProcessingNotes && (
          <div className="mx-4 mb-4 p-2 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                <Paperclip className="inline-block w-4 h-4 mr-1" />
                Attached Files ({attachedFiles.length})
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setAttachedFiles([])}
                className="text-gray-500 hover:text-gray-700"
              >
                Clear all
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {attachedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-full border border-gray-200 text-sm"
                >
                  <span className="text-gray-600">{file.file_name}</span>
                  <button
                    onClick={() => {
                      const newFiles = [...attachedFiles];
                      newFiles.splice(index, 1);
                      setAttachedFiles(newFiles);
                    }}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      {isProcessingFiles && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 bg-blue-500 text-white px-6 py-3 rounded-full shadow-lg z-50">
          <div className="flex items-center">
            <Loader2 className="animate-spin mr-3 h-5 w-5" />
            <span className="font-medium">{processingOperation || 'Processing files...'}</span>
          </div>
        </div>
      )}
    </>
  )
}

export default Component