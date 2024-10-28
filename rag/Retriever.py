import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class Retriever(nn.Module):
    def __init__(self, embedding_model_name, embedding_dim=768):
        super(Retriever, self).__init__()
        # Load the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = embedding_dim
        # Initialize FAISS index for retrieval
        self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def encode_documents(self, documents):
        """Encode a batch of documents using the embedding model."""
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
        return embeddings

    def build_index(self, corpus_embeddings):
        """Add the corpus embeddings to the FAISS index."""
        self.index.reset()
        self.index.add(corpus_embeddings.cpu().detach().numpy())

    def retrieve(self, query, k=5):
        """Retrieve top-k documents based on the query."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

    def forward(self, queries, positive_docs):
        """Calculate the loss between the queries and the positive documents."""
        # Encode the queries and positive documents
        query_embeddings = self.encode_documents(queries)
        positive_embeddings = self.encode_documents(positive_docs)
        
        # Compute the L2 distance loss between query and positive document embeddings
        loss = nn.MSELoss()(query_embeddings, positive_embeddings)
        return loss

    def train_retriever(self, train_loader, optimizer, num_epochs=3):
        """Train the retriever using the WikiQA dataset."""
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                queries, positive_docs, _ = batch
                
                # Calculate loss
                loss = self(queries, positive_docs)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def test_retriever(self, test_loader, k=5):
        """Test the retriever using the WikiQA dataset."""
        self.eval()
        correct_retrievals = 0

        with torch.no_grad():
            for batch in test_loader:
                queries, positive_docs, corpus = batch
                for i in range(len(queries)):
                    query = queries[i]
                    positive_document = positive_docs[i]

                    # Build the index with the corpus documents
                    corpus_embeddings = self.encode_documents(corpus)
                    self.build_index(corpus_embeddings)

                    # Retrieve top-k documents for the query
                    retrieved_indices = self.retrieve(query, k=k)
                    retrieved_docs = [corpus[j] for j in retrieved_indices]

                    if positive_document in retrieved_docs:
                        correct_retrievals += 1

        accuracy = correct_retrievals / len(test_loader)
        print(f"Retrieval Accuracy: {accuracy:.4f}")
        return accuracy
