import os
import torch
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
import PyPDF2
import re
from langchain.document_loaders import TextLoader, UnstructuredFileLoader

def save_model(model, save_directory="saved_model"):
    """Save the trained retriever and generator models."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save retriever (BERT)
    retriever_dir = os.path.join(save_directory, "retriever")
    model.retriever.save_pretrained(retriever_dir)
    model.tokenizer_retriever.save_pretrained(retriever_dir)

    # Save generator (BART)
    generator_dir = os.path.join(save_directory, "generator")
    model.generator.save_pretrained(generator_dir)
    model.tokenizer_generator.save_pretrained(generator_dir)
    print(f"Model saved to {save_directory}")


def evaluate_rag_model(model, test_queries, documents, target_outputs):
    """Evaluate the RAG model on a set of test queries."""
    model.eval()
    
    # Load evaluation metrics using the new 'evaluate' library
    rouge = load("rouge")
    bleu_scores = []

    generated_outputs = []
    for query, target_output in zip(test_queries, target_outputs):
        with torch.no_grad():
            generated_output = model(query, documents)
            generated_outputs.append(generated_output)

            # Calculate BLEU score for the generated output
            bleu_score = sentence_bleu([target_output.split()], generated_output.split())
            bleu_scores.append(bleu_score)

            # Add the results to ROUGE metric
            rouge.add(prediction=generated_output, reference=target_output)
    
    # Compute average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    # Compute ROUGE scores
    rouge_result = rouge.compute()

    print("Evaluation Results:")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")
    print(f"ROUGE Score: {rouge_result}")

    return generated_outputs, avg_bleu_score, rouge_result


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, filename='checkpoint.pth'):
    """Save the model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_dir, filename='checkpoint.pth'):
    """Load the model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    else:
        return 0
    
def PDFLoader(file_path: str):
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
            return chunks
    except Exception as e:
        print(f"[ERROR] Error processing PDF file: {e}")
        return []
    
def load_documents(path):
    """
    Recursively load all supported documents from the note_folder_path.

    :return: List of loaded documents.
    """
    loaders = []
    for root, _, files in os.walk(path):
        print(f"[INFO] Loading documents from {root}")
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext == '.txt' or ext == '.md':
                    loaders.append(TextLoader(file_path, encoding='utf-8'))
                elif ext == '.pdf':
                    loaders.append(PDFLoader(file_path))
                else:
                    # Use a generic loader for unsupported types
                    loaders.append(UnstructuredFileLoader(file_path))
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")

    # Load all documents
    documents = []
    for loader in loaders:
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"[INFO] Loaded {len(loaded_docs)} documents from {loader}")
        except Exception as e:
            print(f"[ERROR] Error loading documents with {loader}: {e}")

    print(f"[INFO] Loaded {len(documents)} documents.")
    return documents