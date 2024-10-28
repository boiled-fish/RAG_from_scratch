import os
import sys

# Get the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
import time
from logging import getLogger
from utils.custom_logger import CustomLogger
from utils.rag_utils import save_model, evaluate_rag_model
from tqdm import tqdm

# Set the custom directory for dataset downloads
os.environ['HF_DATASETS_CACHE'] = './datasets'

# Initialize the CustomLogger
logger = CustomLogger(log_dir_base="../log/rag_logs")
log_dir = logger.get_log_dir()

# Use the standard logging instance
logging = logger.get_logger()

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class RAGModel(nn.Module):
    def __init__(self, retriever_model_name='bert-base-uncased', generator_model_name='facebook/bart-large'):
        super(RAGModel, self).__init__()
        logging.info("Initializing RAGModel with retriever and generator.")

        # Initialize the retriever (using BERT for encoding)
        self.tokenizer_retriever = BertTokenizer.from_pretrained(retriever_model_name)
        self.retriever = BertModel.from_pretrained(retriever_model_name)
        
        # Initialize the generator (using BART for sequence generation)
        self.tokenizer_generator = BartTokenizer.from_pretrained(generator_model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(generator_model_name)

        # Move the model components to the specified device
        self.retriever.to(device)
        self.generator.to(device)

        logging.info("Model initialization complete.")
    
    def encode_query(self, query):
        """Encode the query to generate vector representation."""
        logging.debug(f"Encoding query: {query}")
        inputs = self.tokenizer_retriever(query, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = self.retriever(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Representation of [CLS] token

    def retrieve_documents(self, query_vector, documents, top_k=5):
        """Retrieve the top K documents most relevant to the query."""
        logging.debug("Retrieving documents.")
        doc_vectors = [self.encode_query(doc) for doc in documents]
        similarities = [torch.cosine_similarity(query_vector, doc_vec, dim=-1).item() for doc_vec in doc_vectors]
        top_docs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc[0] for doc in top_docs]

    def generate_output(self, query, documents):
        """Generate the final output sequence."""
        logging.debug("Generating output sequence.")
        # Concatenate the query with the retrieved documents
        input_text = query + " ".join(documents)
        inputs = self.tokenizer_generator(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Use the generator to produce the output
        generated_ids = self.generator.generate(inputs["input_ids"], max_length=50, num_beams=5)
        output = self.tokenizer_generator.decode(generated_ids[0], skip_special_tokens=True)
        return output

    def forward(self, query, documents):
        """Forward pass of the RAG model, including retrieval and generation."""
        logging.debug(f"Performing forward pass for query: {query}")
        query_vector = self.encode_query(query)
        retrieved_docs = self.retrieve_documents(query_vector, documents, top_k=5)
        generated_output = self.generate_output(query, retrieved_docs)
        return generated_output

    def calculate_loss(self, query, target_output, documents, criterion):
        """Calculate the joint loss for retriever and generator."""
        logging.debug("Calculating loss.")
        query_vector = self.encode_query(query)
        retrieved_docs = self.retrieve_documents(query_vector, documents, top_k=5)
        input_text = query + " ".join(retrieved_docs)
        inputs = self.tokenizer_generator(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        labels = self.tokenizer_generator(target_output, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
        outputs = self.generator(input_ids=inputs["input_ids"], labels=labels)
        return outputs.loss


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, filename='checkpoint.pth'):
    """Save the model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_dir, filename='checkpoint.pth'):
    """Load the model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0

def train_rag_model(model, queries, documents, target_outputs, learning_rate=1e-5, epochs=3, checkpoint_interval=1, resume=False):
    """Train the RAG model with checkpointing support."""
    logging.info("Starting training.")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer_generator.pad_token_id)

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))  # TensorBoard writer

    # If resuming, load the checkpoint
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, log_dir)

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()  # Record the start time of the epoch
        total_loss = 0
        logging.info(f"Epoch {epoch + 1} started.")

        for i, (query, target_output) in enumerate(tqdm(zip(queries, target_outputs), total=len(queries), desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()
            loss = model.calculate_loss(query, target_output, documents, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:  # Log every 100 steps
                progress = (i + 1) / len(queries)
                logging.info(f"Epoch {epoch + 1}, Step {i} : Loss = {loss.item():.4f}")
                writer.add_scalar("Training Loss", loss.item(), epoch * len(queries) + i)  # TensorBoard logging

        average_loss = total_loss / len(queries)
        logging.info(f"Epoch {epoch + 1} complete. Average Loss: {average_loss:.4f}")
        writer.add_scalar("Average Epoch Loss", average_loss, epoch + 1)

        # Record the time taken for this epoch
        epoch_duration = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")
        writer.add_scalar("Epoch Duration", epoch_duration, epoch + 1)

        # Save the checkpoint at the end of each epoch or at a set interval
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, log_dir)

    writer.close()

# training and evaluation using WikiQA dataset
if __name__ == "__main__":
    logging.info("Loaded WikiQA dataset.")
    dataset = load_dataset('wiki_qa')

    train_data = dataset['train']
    train_data = train_data.select(range(len(train_data) // 10))

    queries = [entry['question'] for entry in train_data]
    target_outputs = [entry['answer'] for entry in train_data]
    documents = target_outputs[:100]

    model = RAGModel().to(device)  # Move the model to the GPU device if available
    print("Training RAG model with WikiQA dataset...")
    
    # Call training function with the option to resume from a checkpoint
    train_rag_model(model, queries, documents, target_outputs, learning_rate=1e-5, epochs=5, checkpoint_interval=1, resume=True)
    
    print("Saving the trained model...")
    save_model(model, save_directory="rag_trained_model_wikiqa")
    
    print("Evaluating the model...")
    evaluate_rag_model(model, queries, documents, target_outputs)
    
    logging.info("Script finished.")


