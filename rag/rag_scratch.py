import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
import faiss
from evaluate import load
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.tensorboard import SummaryWriter
from utils.custom_logger import CustomLogger

class ScratchRAGModel(nn.Module):
    def __init__(self, log_dir = './log', retriever_model_name='bert-base-uncased', generator_model_name='facebook/bart-large', args=None):
        super(ScratchRAGModel, self).__init__()
        # Initialize the CustomLogger
        self.logger = CustomLogger(log_dir_base=log_dir, logger_name="scratch_rag")
        self.log_dir = self.logger.get_log_dir()

        # Use a separate logger to avoid overwriting the logging module
        self.log = self.logger.get_logger()

        # Check if GPU is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"Using device: {self.device}")

        self.log.info("Initializing RAGModel with retriever and generator.")

        # Initialize the retriever (using BERT for encoding)
        self.top_k = 5
        self.tokenizer_retriever = BertTokenizer.from_pretrained(retriever_model_name, cache_dir='./hugging_face_models')
        self.retriever = BertModel.from_pretrained(retriever_model_name, cache_dir='./hugging_face_models')
        
        # Initialize the generator (using BART for sequence generation)
        self.tokenizer_generator = BartTokenizer.from_pretrained(generator_model_name, cache_dir='./hugging_face_models')
        self.generator = BartForConditionalGeneration.from_pretrained(generator_model_name, cache_dir='./hugging_face_models')

        # Move the model components to the specified device
        self.retriever.to(self.device)
        self.generator.to(self.device)

        # Initialize Faiss index and document storage
        self.faiss_index = None
        self.document_texts = []  # Store the text of documents based on index positions

        self.model = None
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                self.log.warning(f"Parameter {name} does not require grad and will not be trained.")

        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True

        # Verify that all parameters are trainable
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log.info(f"Total parameters: {total_params}")
        self.log.info(f"Trainable parameters: {trainable_params}")

        self.log.info("Model initialization complete.")

    def embed_documents(self, candidate_docs):
        """
        Embed all candidate documents and store them in a Faiss index for fast retrieval.

        Args:
            candidate_docs (list of str): List of candidate documents to embed.
        """
        # Create an empty list to store embeddings
        embeddings = []

        # Embed each document using the retriever model
        self.log.debug(f"Embedding {len(candidate_docs)} documents...")
        for doc in tqdm(candidate_docs, desc="Embedding documents"):
            doc_tokens = self.tokenizer_retriever(doc, return_tensors='pt', padding=True, truncation=True).to(self.device) 
            with torch.no_grad():
                doc_embedding = self.retriever(**doc_tokens).last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(doc_embedding)

        # Convert embeddings to a NumPy array and store them in Faiss
        self.log.debug("Stacking embeddings...")
        embeddings = np.vstack(embeddings).astype(np.float32)

        # Initialize a Faiss index
        d = embeddings.shape[1]  # Dimension of the embeddings
        self.log.debug(f"Initializing Faiss index with dimension {d}...")
        self.faiss_index = faiss.IndexFlatL2(d)  # Use L2 distance (or you can choose cosine similarity index)
        self.log.debug("Adding embeddings to Faiss index...")
        self.faiss_index.add(embeddings)  # Add embeddings to the Faiss index

        # Store document texts for retrieval later
        self.document_texts = candidate_docs

    def retrieve_documents(self, query, top_k=5):
        """
        Retrieves top-k most similar documents to the given query using the Faiss index.

        Args:
            query (str): The input query text.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[Tuple[int, str, float]]: A list of tuples containing the index, document, and similarity score.
        """
        # Embed the query using the retriever
        self.log.debug(f"Embedding query: {query}")
        query_tokens = self.tokenizer_retriever(query, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            query_embedding = self.retriever(**query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        # Search the Faiss index to find the top-k most similar documents
        self.log.debug(f"Searching Faiss index for top {top_k} documents...")
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Retrieve the top-k documents and their scores
        self.log.debug(f"Retrieving top {top_k} documents...")
        top_k_docs = [(idx, self.document_texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

        return top_k_docs

    def generate_output(self, query, documents):
        """Generate the final output sequence."""
        self.log.debug("Generating output sequence.")
        # Concatenate the query with the retrieved documents with a space in between
        input_text = query + " " + " ".join(documents)
        inputs = self.tokenizer_generator(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Use the generator to produce the output
        self.log.debug("Generating output ids...")
        generated_ids = self.generator.generate(inputs["input_ids"], max_length=50, num_beams=5)
        self.log.debug("Decoding generated ids to text...")
        output = self.tokenizer_generator.decode(generated_ids[0], skip_special_tokens=True)
        return output

    def forward(self, input_query, max_length=50):
        """
        Forward pass to retrieve documents based on input query and generate a response.

        Args:
            input_query (str): The input query text.
            candidate_docs (list of str): List of candidate documents to retrieve from.
            max_length (int): Maximum length for the generated response.
            top_k (int): Number of top documents to retrieve.

        Returns:
            generated_text (str): The generated response based on the retrieved documents.
        """
        # Step 1: Retrieve top-k documents based on input_query
        self.log.debug(f"Retrieving {self.top_k} documents for query: {input_query}")
        retrieved_docs = self.retrieve_documents(input_query, top_k=self.top_k)

        # Step 2: Concatenate the retrieved documents into a single context string
        self.log.debug(f"Concatenating retrieved documents: {retrieved_docs}")
        context = " ".join([doc[1] for doc in retrieved_docs])  # doc[1] is the document text

        # Step 3: Tokenize the input query and context using the generator's tokenizer
        combined_input = f"{input_query} {context}"
        self.log.debug(f"Tokenizing combined input: {combined_input}")
        input_tokens = self.tokenizer_generator(
            combined_input, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(self.device)

        # Step 4: Pass the combined input to the generator model to generate a response
        self.log.debug("Passing combined input to generator model...")
        with torch.no_grad():
            generated_ids = self.generator.generate(
                input_ids=input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                max_length=max_length,
                num_beams=3,  # Adjust num_beams for more diverse generations
                early_stopping=True
            )

        # Decode the generated ids to text
        self.log.debug("Decoding generated ids to text...")
        generated_text = self.tokenizer_generator.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text

    def calculate_loss(self, query, target_output):
        """Calculate the joint loss for retriever and generator."""
        self.log.debug("Calculating loss.")
        
        self.log.debug(f"Retrieving {self.top_k} documents for query: {query}")
        retrieved_docs = self.retrieve_documents(query, top_k=self.top_k)
        
        self.log.debug(f"Concatenating query and retrieved documents: {query} {retrieved_docs}")
        input_text = query + " " + " ".join([doc[1] for doc in retrieved_docs])
        inputs = self.tokenizer_generator(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True).to(self.device)
        
        self.log.debug(f"Tokenizing target output: {target_output}")
        labels = self.tokenizer_generator(
            target_output,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=512
        )["input_ids"].to(self.device)
        labels[labels == self.tokenizer_generator.pad_token_id] = -100

        self.log.debug("Passing inputs to generator model...")
        outputs = self.generator(input_ids=inputs["input_ids"], labels=labels)
        return outputs.loss
    
    def evaluate(self, queries, target_outputs):
        """Evaluate the model on a set of queries."""
        self.retriever.eval()
        self.generator.eval()

        rouge = load("rouge")
        bleu_scores = []

        self.log.info(f"Evaluating {len(queries)} queries...")
        for query, target_output in tqdm(zip(queries, target_outputs), total=len(queries), desc="Evaluating"):
            with torch.no_grad():
                generated_output = self.forward(query)

                # Calculate BLEU score for the generated output
                bleu_score = sentence_bleu([target_output.split()], generated_output.split())
                bleu_scores.append(bleu_score)

                # Add the results to ROUGE metric
                rouge.add(prediction=generated_output, reference=target_output)

        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        self.log.info(f"Average BLEU Score: {avg_bleu_score:.4f}")

        rouge_result = rouge.compute()
        self.log.info(f"ROUGE Score: {rouge_result}")

        return avg_bleu_score, rouge_result

    def save_model(self, save_directory):
        """Saves the retriever and generator models and tokenizers to a specified directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        retriever_path           = os.path.join(save_directory, 'retriever_model')
        generator_path           = os.path.join(save_directory, 'generator_model')
        tokenizer_retriever_path = os.path.join(save_directory, 'tokenizer_retriever')
        tokenizer_generator_path = os.path.join(save_directory, 'tokenizer_generator')

        self.log.info("Saving retriever model and tokenizer...")
        self.retriever.save_pretrained(retriever_path)
        self.tokenizer_retriever.save_pretrained(tokenizer_retriever_path)

        self.log.info("Saving generator model and tokenizer...")
        self.generator.save_pretrained(generator_path)
        self.tokenizer_generator.save_pretrained(tokenizer_generator_path)
        
        self.log.info(f"Models saved to {save_directory}")

    def load_model(self, load_directory):
        """Loads the retriever and generator models and tokenizers from the specified directory."""
        retriever_path           = os.path.join(load_directory, 'retriever_model')
        generator_path           = os.path.join(load_directory, 'generator_model')
        tokenizer_retriever_path = os.path.join(load_directory, 'tokenizer_retriever')
        tokenizer_generator_path = os.path.join(load_directory, 'tokenizer_generator')
        
        # Load retriever model and tokenizer
        self.log.info("Loading retriever model and tokenizer...")
        self.retriever = BertModel.from_pretrained(retriever_path)
        self.tokenizer_retriever = BertTokenizer.from_pretrained(tokenizer_retriever_path)
        
        # Load generator model and tokenizer
        self.log.info("Loading generator model and tokenizer...")
        self.generator = BartForConditionalGeneration.from_pretrained(generator_path)
        self.tokenizer_generator = BartTokenizer.from_pretrained(tokenizer_generator_path)
        
        self.retriever.to(self.device)
        self.generator.to(self.device)
        
        self.log.info(f"Models loaded from {load_directory}")

    def train_scratch_rag_model(self, queries, target_outputs, learning_rate=1e-5, epochs=3, checkpoint_interval=1):
        """Train the RAG model with checkpointing support."""
        self.log.info("Starting training.")
        writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))  # TensorBoard writer

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(0, epochs):
            epoch_start_time = time.time()  # Record the start time of the epoch
            total_loss = 0
            self.log.info(f"Epoch {epoch + 1} started.")

            for i, (query, target_output) in enumerate(tqdm(zip(queries, target_outputs), total=len(queries), desc=f"Epoch {epoch + 1}")):
                optimizer.zero_grad()
                loss = self.calculate_loss(query, target_output)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 100 == 0:  # Log every 100 steps
                    self.log.info(f"Epoch {epoch + 1}, Step {i} : Loss = {loss.item():.4f}")
                    writer.add_scalar("Training Loss", loss.item(), epoch * len(queries) + i)  # TensorBoard logging

            average_loss = total_loss / len(queries)
            self.log.info(f"Epoch {epoch + 1} complete. Average Loss: {average_loss:.4f}")
            writer.add_scalar("Average Epoch Loss", average_loss, epoch + 1)

            # Record the time taken for this epoch
            epoch_duration = time.time() - epoch_start_time
            self.log.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")
            writer.add_scalar("Epoch Duration", epoch_duration, epoch + 1)

            # Save the checkpoint at the end of each epoch or at a set interval
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(
                    save_directory=os.path.join(self.log_dir, f"checkpoint_{epoch + 1}")
                )

        writer.close()

    def inference(self, query):
        self.retriever.eval()
        self.generator.eval()

        corpus_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', "text-corpus", cache_dir='./datasets')

        documents = [entry['passage'] for entry in corpus_dataset['passages']]
        self.embed_documents(documents)

        return self.forward(query)
    
    # Training and evaluation using WikiQA dataset
    def pipeline(self, 
                 num_train_epochs=3, 
                 learning_rate=1e-5, 
                 save_model=False,
                 checkpoint_interval=1
        ):
        self.log.info("Loading rag-mini-wikipedia dataset...")
        #dataset = load_dataset('wiki_qa', cache_dir='./datasets')
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
        print(len(train_data))
        train_data = train_data.select(range(len(train_data)))

        queries = [entry['question'] for entry in train_data]
        target_outputs = [entry['answer'] for entry in train_data]
        documents = [entry['passage'] for entry in corpus_dataset['passages']]
        self.embed_documents(documents)

        # Call training function with the option to resume from a checkpoint
        self.log.info("Training RAG model with WikiQA dataset...")
        self.train_scratch_rag_model(
            queries, target_outputs, 
            learning_rate=learning_rate, 
            epochs=num_train_epochs, 
            checkpoint_interval=checkpoint_interval
        )
        
        self.log.info("Saving the trained model...")
        self.save_model(save_directory="rag_trained_model_wikiqa")
        
        self.log.info("Evaluating the model...")
        self.evaluate(queries, target_outputs)
        
        if save_model:
            save_pth = os.path.join(self.log_dir, "rag_trained_model_wikiqa")
            self.save_model(save_directory=save_pth)

        self.log.info("Pipeline finished.")

if __name__ == "__main__":
    log_dir = './logs'
    model = ScratchRAGModel(log_dir)
    model.pipeline(
        num_train_epochs = 3,
        learning_rate = 1e-5,
        save_model = True
    )

    query = "What is the capital of France?"
    model.forward(query)