import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from evaluate import load
from datasets import load_dataset
from utils.custom_logger import CustomLogger
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from tqdm import tqdm

class WikiQADataset(Dataset):
    """Custom Dataset for WikiQA data."""
    def __init__(self, queries, positive_docs):
        self.queries = queries
        self.positive_docs = positive_docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.positive_docs[idx]

class Retriever(nn.Module):
    def __init__(self, log_dir='./log', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', test_size=0.2, load_dataset_ratio=1.0, device=None, cache_dir=None):
        super(Retriever, self).__init__()
        self.logger = CustomLogger(log_dir_base=log_dir, logger_name="retriever_logs_30000Epochs")
        self.log_dir = self.logger.get_log_dir()

        # Use a separate logger to avoid overwriting the logging module
        self.log = self.logger.get_logger()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.log.info(f"Initializing Retriever on device: {self.device}")

        # TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)  # Initialize TensorBoard writer

        # Load the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, cache_dir=cache_dir)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, cache_dir=cache_dir)

        self.embedding_model.to(self.device)
        self.embedding_dim = self.embedding_model.config.hidden_size

        self.log.info(f"Loaded embedding model: {embedding_model_name} with embedding dimension: {self.embedding_dim}")

        # Initialize FAISS index for retrieval
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.log.info(f"Initialized FAISS Index with dimension: {self.embedding_dim}")

        qa_dataset = load_dataset(
            'rag-datasets/rag-mini-wikipedia',
            'question-answer',
            cache_dir='./datasets'
        )
        corpus_dataset = load_dataset(
            'rag-datasets/rag-mini-wikipedia',
            'text-corpus',
            cache_dir='./datasets'
        )
        train_data = qa_dataset['test']
        train_data = train_data.select(range(int(len(train_data) * load_dataset_ratio)))

        self.queries = [entry['question'] for entry in train_data]
        self.positive_docs = [entry['answer'] for entry in train_data]
        self.log.info(f"Loaded {len(self.queries)} queries and positive documents.")

        self.corpus = [entry['passage'] for entry in corpus_dataset['passages']]

        # Encode corpus with specified batch_size to prevent OOM errors
        self.corpus_embeddings = self.encode_corpus(self.corpus, batch_size=64)
        self.log.info(f"Encoded corpus with {len(self.corpus)} passages.")

        # build the FAISS index
        self.build_index(self.corpus_embeddings)
        self.log.info(f"Built FAISS index with {self.corpus_embeddings.shape[0]} embeddings.")

        # Split data into training and testing sets
        train_queries, test_queries, train_docs, test_docs = train_test_split(self.queries, self.positive_docs, test_size=test_size)

        # Create datasets and dataloaders
        self.train_dataset = WikiQADataset(train_queries, train_docs)
        self.test_dataset = WikiQADataset(test_queries, test_docs)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        self.log.info(f"Created DataLoaders with train size: {len(self.train_loader)} and test size: {len(self.test_loader)}.")

    def encode_documents(self, documents):
        """Encode a batch of documents using the embedding model."""
        # Tokenize the documents
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.  items()}
        # Compute embeddings
        outputs = self.embedding_model(**inputs)
        # Get the embeddings, e.g., use mean pooling
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_mask = inputs['attention_mask']  # [batch_size, seq_len]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_embeddings / sum_mask  # [batch_size, hidden_size]
        return embeddings

    def encode_corpus(self, corpus, batch_size=64):
        """Encode the entire corpus."""
        all_embeddings = []
        self.embedding_model.eval()
        with torch.no_grad():
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i:i+batch_size]
                embeddings = self.encode_documents(batch)
                all_embeddings.append(embeddings.cpu())
                self.log.debug(f"Encoded batch {i // batch_size + 1}/{(len(corpus) - 1) // batch_size + 1}")
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def build_index(self, corpus_embeddings):
        """Add the corpus embeddings to the FAISS index."""
        self.index.reset()
        embeddings_numpy = corpus_embeddings.cpu().detach().numpy()
        self.index.add(embeddings_numpy)
        self.log.info(f"Built FAISS index with {embeddings_numpy.shape[0]} embeddings.")

    def retrieve(self, queries, k=1):
        """Retrieve top-k documents for a batch of queries."""
        # Compute query embeddings
        self.embedding_model.eval()
        with torch.no_grad():
            embeddings = self.encode_documents(queries)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            query_embeddings = embeddings.cpu().detach().numpy()
            # Search the index
            _, indices = self.index.search(query_embeddings, k)
        self.log.debug(f"Retrieved top {k} documents for batch of queries.")

        # Retrieve the actual documents using the indices
        positive_docs = []
        for idx_list in indices:
            # For each query, take the first retrieved document as the positive document
            doc_idx = idx_list[0]
            positive_doc = self.corpus[doc_idx]
            positive_docs.append(positive_doc)
        return positive_docs

    def forward(self, queries, target_docs):
        """Calculate the loss between the queries and the positive documents using in-batch negatives."""
        # Retrieve positive documents for the batch of queries
        positive_docs = self.retrieve(queries)

        # Encode the queries and positive documents
        doc_embeddings = self.encode_documents(positive_docs)  # Shape: (batch_size, embedding_dim)
        target_doc_embeddings = self.encode_documents(target_docs)  # Shape: (batch_size, embedding_dim)

        # Normalize embeddings to compute cosine similarity
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        target_doc_embeddings = F.normalize(target_doc_embeddings, p=2, dim=1)

        # Compute similarity matrix between queries and documents
        similarity_matrix = torch.matmul(target_doc_embeddings, doc_embeddings.T)  # Shape: (batch_size, batch_size)

        # Labels are indices from 0 to batch_size - 1
        labels = torch.arange(doc_embeddings.size(0)).to(self.device)

        # Use CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(similarity_matrix, labels)
        return loss

    def train_retriever(self, num_epochs=3, resume=False, start_epoch=0):
        """Train the retriever using the training data."""
        self.train()
        
        if resume:
            self.log.info(f"Resuming training for {num_epochs} epochs.")
        else:
            self.log.info(f"Starting training for {num_epochs} epochs.")

        if resume:
            self.load_model(os.path.join(self.log_dir, f"retriever_model"))
        else:
            start_epoch = 0

        self.log.info(f"Training retriever with {num_epochs} epochs.")
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Epochs"):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                queries, positive_docs = batch
                queries = list(queries)

                # Calculate loss
                loss = self(queries, positive_docs)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.writer.add_scalar('Loss/batch', loss.item(), batch_idx)

            avg_loss = total_loss / len(self.train_loader)
            if (epoch + 1) % 500 == 0:
                self.log.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Log the average epoch loss to TensorBoard
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)

            if (epoch + 1) % 100 == 0:
                save_directory = os.path.join(self.log_dir, f"retriever_model")
                self.save_model(save_directory)

    def test_retriever(self, k=5):
        """Test the retriever using the test data."""
        self.eval()
        correct_retrievals = 0
        total_queries = 0

        self.log.info(f"Testing retriever with k={k}")

        for batch in tqdm(self.test_loader, desc="Testing"):
            queries, positive_docs = batch
            queries = list(queries)
            positive_docs = list(positive_docs)
            total_queries += len(queries)

            # Retrieve top-k documents for the batch of queries
            retrieved_docs_batch = self.retrieve(queries, k=k)

            for i in range(len(queries)):
                positive_document = positive_docs[i]
                retrieved_docs = retrieved_docs_batch[i]

                if positive_document in retrieved_docs:
                    correct_retrievals += 1

        accuracy = correct_retrievals / total_queries
        self.log.info(f"Retrieval Accuracy: {accuracy:.4f}")
        # Log accuracy to TensorBoard
        self.writer.add_scalar('Accuracy/test', accuracy, 0)
        return accuracy

    def save_model(self, save_directory):
        self.embedding_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        idx_file_path = os.path.join(save_directory, 'retriever_index.faiss')
        faiss.write_index(self.index, idx_file_path)
        self.log.info(f"Model saved at {save_directory}")

    def load_model(self, save_directory):
        self.embedding_model.from_pretrained(save_directory)
        self.tokenizer.from_pretrained(save_directory)
        idx_file_path = os.path.join(save_directory, 'retriever_index.faiss')
        self.index = faiss.read_index(idx_file_path)
        self.log.info(f"Model loaded from {save_directory}")

    def inference(self, query, top_k=5):
        self.eval()
        # Build the FAISS index if not already built
        if self.index.ntotal == 0:
            self.build_index(self.corpus_embeddings)

        retrieved_indices = self.retrieve(query, k=top_k)
        retrieved_docs = [self.corpus[j] for j in retrieved_indices]

        self.log.info(f"Inference completed for query: {query}")
        return list(retrieved_docs)

    def pipeline(self, learning_rate=1e-5, num_epochs=3, resume=False):
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Log the learning rate to TensorBoard
        self.writer.add_scalar('Hyperparameters/Learning Rate', learning_rate, 0)
        self.train_retriever(num_epochs=num_epochs, resume=resume)

        # Save the retriever model
        self.log.info("Saving retriever model")
        save_directory = os.path.join(self.log_dir, f"retriever_model")
        self.save_model(save_directory)

        # Test the retriever
        self.log.info("Testing retriever")
        self.test_retriever()

        # Close the TensorBoard writer
        self.writer.close()

# Usage example
if __name__ == "__main__":    
    retriever = Retriever(
        log_dir='./log',
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_dir='./hugging_face_models'
    )
    retriever.test_retriever()
