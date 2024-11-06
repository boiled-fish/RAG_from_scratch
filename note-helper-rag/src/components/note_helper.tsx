'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Upload, Send, Bot, User, Folder } from 'lucide-react'

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
    file_content: string[]
  }>
}

function Component() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState('Ollama')
  const [attachedFiles, setAttachedFiles] = useState<Array<{
    file_name: string,
    file_content: string[]
  }>>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showFolderDialog, setShowFolderDialog] = useState(true)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const [isModelInitialized, setIsModelInitialized] = useState(false)
  const [isProcessingNotes, setIsProcessingNotes] = useState(false)
  const [manualPath, setManualPath] = useState('')
  const [showManualInput, setShowManualInput] = useState(false)

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

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
  
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            message: input, 
            model: model.toLowerCase(),
            files: attachedFiles
          }),
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
    const files = Array.from(e.target.files || [])
    const processedFiles = await Promise.all(files.map(async file => {
      const content = await file.text()
      return {
        file_name: file.name,
        file_content: content.split('\n')
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
        <DialogContent className="bg-white" onPointerDownOutside={(e) => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle className="text-black">Select Notes Folder</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            {!showManualInput ? (
              <>
                <p className="text-black">Please choose how you want to specify the notes folder:</p>
                <div className="flex flex-col gap-2">
                  <Button 
                    variant="outline" 
                    onClick={handleUseDefault}
                    className="text-black hover:bg-black hover:text-white transition-colors"
                  >
                    Use Default
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => setShowManualInput(true)}
                    className="text-black hover:bg-black hover:text-white transition-colors"
                  >
                    Enter Path Manually
                  </Button>
                </div>
              </>
            ) : (
              <>
                <p className="text-black">Enter the absolute path to your notes folder:</p>
                <Input
                  type="text"
                  placeholder="e.g., C:\Users\YourName\Documents\Notes"
                  value={manualPath}
                  onChange={(e) => setManualPath(e.target.value)}
                  className="mb-2"
                />
                <div className="flex justify-between gap-2">
                  <Button 
                    onClick={handleManualPathSubmit}
                    className="flex-1 text-black hover:bg-black hover:text-white transition-colors"
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
                    className="flex-1 text-black hover:bg-black hover:text-white transition-colors"
                  >
                    Back
                  </Button>
                </div>
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>

      <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
        <Card className="m-4">
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <h1 className="text-2xl font-bold">Note Helper</h1>
              <Select 
                value={model} 
                onValueChange={setModel} 
                disabled={isProcessingNotes}
              >
                <SelectTrigger className={`w-[180px] ${isProcessingNotes ? 'opacity-50 cursor-not-allowed' : ''}`}>
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

        <ScrollArea className="flex-grow mx-4 mb-4 p-4 rounded-xl bg-white dark:bg-gray-800" ref={scrollAreaRef}>
          {messages.map((message, index) => (
            <div key={index} className={`mb-4 ${message.isUser ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block p-3 rounded-xl ${message.isUser ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700'}`}>
                <div className="flex items-center mb-2">
                  {message.isUser ? <User className="mr-2" /> : <Bot className="mr-2" />}
                  <span className="font-bold">{message.isUser ? 'You' : 'AI'}</span>
                </div>
                <p>{message.text}</p>
                {message.files && message.files.length > 0 && (
                  <div className="mt-2 text-sm">
                    <p>Attached files:</p>
                    <ul>
                      {message.files.map((file, fileIndex) => (
                        <li key={fileIndex}>{file.file_name}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="text-center">
              <p>Assistant is thinking...</p>
            </div>
          )}
        </ScrollArea>

        <div className="flex items-center space-x-2 m-4">
          <Input
            type="text"
            placeholder={isProcessingNotes ? "Please wait while processing notes..." : "Type your message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            className="flex-grow rounded-xl"
            disabled={isProcessingNotes || isLoading}
          />
          <label htmlFor="file-upload" className={`cursor-pointer ${isProcessingNotes || isLoading ? 'opacity-50 pointer-events-none' : ''}`}>
            <Upload className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
          </label>
          <input
            id="file-upload"
            type="file"
            multiple
            className="hidden"
            onChange={handleFileAttach}
            disabled={isProcessingNotes || isLoading}
          />
          <Button 
            onClick={handleSend} 
            disabled={isProcessingNotes || isLoading}
            className={isProcessingNotes || isLoading ? 'opacity-50' : ''}
          >
            <Send className="mr-2 h-4 w-4" /> Send
          </Button>
        </div>
        {attachedFiles.length > 0 && !isProcessingNotes && (
          <div className="mx-4 mb-4 text-sm text-gray-500 dark:text-gray-400">
            Attached: {attachedFiles.map(f => f.file_name).join(', ')}
          </div>
        )}
      </div>
    </>
  )
}

export default Component