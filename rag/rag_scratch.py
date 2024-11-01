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
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from torchvision import transforms

class ScratchRAGModel(nn.Module):
    def __init__(self, log_dir = './log', retriever_model_name='bert-base-uncased', generator_model_name='facebook/bart-large', load_dataset_ratio=1.0, args=None):
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
        self.document_embeddings = None

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
        train_data = train_data.select(range(int(len(train_data) * load_dataset_ratio)))

        total_queries = [entry['question'] for entry in train_data]
        total_target_outputs = [entry['answer'] for entry in train_data]

        self.train_queries = total_queries[:int(len(total_queries) * 0.8)]
        self.train_target_outputs = total_target_outputs[:int(len(total_target_outputs) * 0.8)]
        self.test_queries = total_queries[int(len(total_queries) * 0.8):]
        self.test_target_outputs = total_target_outputs[int(len(total_target_outputs) * 0.8):]

        self.documents = [entry['passage'] for entry in corpus_dataset['passages']]
        self.embed_documents(self.documents)

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
        embeddings_list = []

        # Embed each document using the retriever model
        self.log.debug(f"Embedding {len(candidate_docs)} documents...")
        for doc in tqdm(candidate_docs, desc="Embedding documents"):
            doc_tokens = self.tokenizer_retriever(doc, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                doc_embedding = self.retriever(**doc_tokens).last_hidden_state[:, 0, :]
            embeddings_list.append(doc_embedding.cpu())

        # Convert embeddings to a NumPy array and store them in Faiss
        self.log.debug("Stacking embeddings...")
        self.document_embeddings = torch.vstack(embeddings_list)

        embeddings_np = self.document_embeddings.numpy().astype(np.float32)

        # Initialize a Faiss index
        d = embeddings_np.shape[1]  # Dimension of the embeddings
        self.log.debug(f"Initializing Faiss index with dimension {d}...")
        self.faiss_index = faiss.IndexFlatL2(d)  # Use L2 distance (or you can choose cosine similarity index)
        self.log.debug("Adding embeddings to Faiss index...")
        self.faiss_index.add(embeddings_np)  # Add embeddings to the Faiss index

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
            query_embedding = self.retriever(**query_tokens).last_hidden_state.mean(dim=1).detach().cpu().numpy()

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

    def get_target_doc_index(self, query, target_output):
        """
        Finds the index of the target document in self.document_texts.

        Args:
            query (str): The input query text.
            target_output (str): The expected output text.

        Returns:
            int: Index of the target document in self.document_texts.
        """
        # Option 1: Use a ground-truth mapping if available
        if hasattr(self, 'query_to_doc_index_mapping'):
            target_doc_index = self.query_to_doc_index_mapping.get(query)
            if target_doc_index is not None:
                return target_doc_index

        # Option 2: Find the document most similar to the target_output
        # Embed the target output using the retriever
        target_tokens = self.tokenizer_retriever(
            target_output, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            target_embedding = self.retriever(**target_tokens).last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)

        # Normalize embeddings
        target_embedding = target_embedding / target_embedding.norm(dim=1, keepdim=True)
        doc_embeddings = self.document_embeddings.to(self.device)
        doc_embeddings_norm = doc_embeddings / doc_embeddings.norm(dim=1, keepdim=True)

        # Compute cosine similarities
        similarities = torch.matmul(target_embedding, doc_embeddings_norm.T)  # Shape: (1, num_docs)

        # Get the index of the most similar document
        target_doc_index = torch.argmax(similarities, dim=1).item()
        return target_doc_index

    def calculate_loss(self, query, target_output, criterion):
        """Calculate the joint loss for retriever and generator."""
        self.log.debug("Calculating loss.")

        # Step 1: Embed the query
        query_tokens = self.tokenizer_retriever(query, return_tensors='pt', padding=True, truncation=True).to(self.device)
        query_embedding = self.retriever(**query_tokens).last_hidden_state.mean(dim=1)

        # Step 2: Compute similarities with all documents
        doc_embeddings = self.document_embeddings.to(self.device)  # Precomputed document embeddings

        # Step 3: normalize embeddings
        query_norm = query_embedding / query_embedding.norm(dim=1, keepdim=True)
        doc_norm = doc_embeddings / doc_embeddings.norm(dim=1, keepdim=True)

        similarities = torch.matmul(query_norm, doc_norm.T)

        # Step 4: Compute retrieval loss (here using CrossEntropyLoss)
        target_doc_index = self.get_target_doc_index(query, target_output)
        retrieval_labels = torch.tensor([target_doc_index]).to(self.device)
        retrieval_loss = criterion(similarities, retrieval_labels)

        # Step 5: Generate the response using top-k retrieved documents
        _, top_k_indices = torch.topk(similarities, self.top_k, dim=1)
        top_k_indices = top_k_indices.squeeze(0).tolist()
        retrieved_texts = [self.document_texts[i] for i in top_k_indices]
        input_text = query + " " + " ".join(retrieved_texts)
        inputs = self.tokenizer_generator(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # Step 6: Compute generator loss
        labels = self.tokenizer_generator(
            target_output,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=512
        )["input_ids"].to(self.device)
        labels[labels == self.tokenizer_generator.pad_token_id] = -100
        outputs = self.generator(input_ids=inputs["input_ids"], labels=labels)
        generator_loss = outputs.loss

        # Step 7: Compute total loss
        total_loss = retrieval_loss + generator_loss
        return total_loss

    def evaluate(self, top_k_values=[5, 10, 20, 30, 40, 50], writer=None):
      """Evaluate the model on a set of queries."""
      self.retriever.eval()
      self.generator.eval()

      # Initialize evaluation metrics per k
      rouge = load("rouge")
      exact_match_scores_at_k = {}
      recall_scores_at_k = {}
      bleu_scores_at_k = {}
      rouge_l_scores_at_k = {}

      self.log.info(f"Evaluating {len(self.test_queries)} queries...")

      # Loop over each value of k
      for k in top_k_values:
          self.log.info(f"Evaluating for K={k}...")
          exact_match_scores = []
          recall_scores = []
          bleu_scores = []
          rouge_l_scores = []

          # Iterate through each query-answer pair
          for query, target_output in tqdm(zip(self.test_queries, self.test_target_outputs),
                                          total=len(self.test_queries), desc=f"Evaluating K={k}"):
              with torch.no_grad():
                  # Step 1: Retrieve top k documents
                  retrieved_docs = self.retrieve_documents(query, top_k=k)
                  retrieved_texts = [doc[1] for doc in retrieved_docs]

                  # Step 2: Check if the correct answer is in the retrieved documents (Recall @ K)
                  recall_at_k = any(target_output in doc for doc in retrieved_texts)
                  recall_scores.append(recall_at_k)

                  # Step 3: Generate output using retrieved documents (top k)
                  generated_output = self.generate_output(query, retrieved_texts)

                  # Step 4: Calculate Exact Match
                  exact_match_score = int(generated_output.strip() == target_output.strip())
                  exact_match_scores.append(exact_match_score)

                  # Step 5: Calculate BLEU score
                  bleu_score = sentence_bleu([target_output.split()], generated_output.split())
                  bleu_scores.append(bleu_score)

                  # Step 6: Calculate ROUGE-L score
                  rouge_l_result = rouge.compute(predictions=[generated_output],
                                                references=[target_output], rouge_types=["rougeL"])
                  rouge_l_scores.append(rouge_l_result["rougeL"])

          # Store metrics for the current k
          exact_match_scores_at_k[k] = exact_match_scores
          recall_scores_at_k[k] = recall_scores
          bleu_scores_at_k[k] = bleu_scores
          rouge_l_scores_at_k[k] = rouge_l_scores

      # Calculate average metrics per k
      avg_exact_match_at_k = {k: np.mean(exact_match_scores_at_k[k]) * 100 for k in top_k_values}
      avg_bleu_score_at_k = {k: np.mean(bleu_scores_at_k[k]) * 100 for k in top_k_values}
      avg_rouge_l_score_at_k = {k: np.mean(rouge_l_scores_at_k[k]) * 100 for k in top_k_values}
      avg_recall_at_k = {k: np.mean(recall_scores_at_k[k]) * 100 for k in top_k_values}

      # Log average metrics
      for k in top_k_values:
          self.log.info(f"K={k}:")
          self.log.info(f"  Average Exact Match Score: {avg_exact_match_at_k[k]:.2f}%")
          self.log.info(f"  Average BLEU Score: {avg_bleu_score_at_k[k]:.2f}%")
          self.log.info(f"  Average ROUGE-L Score: {avg_rouge_l_score_at_k[k]:.2f}%")
          self.log.info(f"  Recall @ {k}: {avg_recall_at_k[k]:.2f}%")

      # Log results to TensorBoard if a writer is provided
      if writer:
          for k in top_k_values:
              writer.add_scalar(f"Evaluation/BLEU_Score@{k}", avg_bleu_score_at_k[k], 100)
              writer.add_scalar(f"Evaluation/Exact_Match@{k}", avg_exact_match_at_k[k], 100)
              writer.add_scalar(f"Evaluation/ROUGE_L_Score@{k}", avg_rouge_l_score_at_k[k], 100)
              writer.add_scalar(f"Evaluation/Recall@{k}", avg_recall_at_k[k], 100)

      self.plot_evaluation_results(
          exact_match=avg_exact_match_at_k,
          avg_recall_at_k=avg_recall_at_k,
          bleu_score=avg_bleu_score_at_k,
          rouge_l_score=avg_rouge_l_score_at_k,
          writer=writer,
          top_k_values=top_k_values
      )

      # Return all computed metrics per k
      return avg_bleu_score_at_k, avg_exact_match_at_k, avg_rouge_l_score_at_k, avg_recall_at_k

    def plot_evaluation_results(self, exact_match, avg_recall_at_k, bleu_score, rouge_l_score, top_k_values, writer=None):
        """
        Plot the evaluation results:
        - Exact Match vs. K Retrieved Docs
        - Recall @ K Retrieved Docs
        - BLEU-1 and ROUGE-L vs. K Retrieved Docs
        """
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: NQ Exact Match
        exact_match_values = [exact_match[k] for k in top_k_values]
        axes[0].plot(top_k_values, exact_match_values, label="Exact Match", color="orange")
        axes[0].set_xlabel("K Retrieved Docs")
        axes[0].set_ylabel("Exact Match (%)")
        axes[0].legend()
        axes[0].set_title("Exact Match vs. Number of Retrieved Documents")

        # Plot 2: Answer Recall @ K
        recall_values = [avg_recall_at_k[k] for k in top_k_values]
        axes[1].plot(top_k_values, recall_values, label="Recall @ K", color="blue")
        axes[1].set_xlabel("K Retrieved Docs")
        axes[1].set_ylabel("Recall @ K (%)")
        axes[1].legend()
        axes[1].set_title("Answer Recall @ K vs. Number of Retrieved Documents")

        # Plot 3: BLEU-1 and ROUGE-L Score
        bleu_values = [bleu_score[k] for k in top_k_values]
        rouge_values = [rouge_l_score[k] for k in top_k_values]
        axes[2].plot(top_k_values, bleu_values, linestyle='dashdot', color="green", label="BLEU-1")
        axes[2].plot(top_k_values, rouge_values, linestyle='dashed', color="red", label="ROUGE-L")
        axes[2].set_xlabel("K Retrieved Docs")
        axes[2].set_ylabel("Score (%)")
        axes[2].legend()
        axes[2].set_title("BLEU-1 and ROUGE-L vs. Number of Retrieved Documents")

        # Layout and display
        plt.tight_layout()

        # Save plot to TensorBoard
        if writer:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image = Image.open(buffer)
            image = transforms.ToTensor()(image)
            writer.add_image("Evaluation Metrics vs. K", image, global_step=100)
            self.log.info(f"Logged evaluation plots to TensorBoard")

        # Save plot to file
        fig.savefig(os.path.join(self.log_dir, "evaluation_results.png"))

    def evaluate_with_posterior_heatmap(self, query, top_k=5, writer=None):
        """
        Evaluate the model by generating output and creating a heatmap to show document posteriors.

        Args:
            query (str): The input query.
            target_output (str, optional): The target output for comparison (not necessary for heatmap).
            top_k (int): Number of top documents to retrieve.

        Returns:
            heatmap_data (np.array): A 2D array containing document posterior probabilities for each generated token.
        """
        # Step 1: Retrieve top-K documents based on the input query
        retrieved_docs = self.retrieve_documents(query, top_k=top_k)
        retrieved_texts = [doc[1] for doc in retrieved_docs]

        # Step 2: Tokenize the input query and retrieved documents
        combined_input = f"{query} " + " ".join(retrieved_texts)
        input_tokens = self.tokenizer_generator(combined_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

        # Step 3: Generate the output sequence while keeping track of posterior probabilities
        with torch.no_grad():
            # Enable output attentions to get the attention scores from the model
            outputs = self.generator.generate(
                input_ids=input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                max_length=50,
                num_beams=1,
                output_attentions=True,  # Get attentions to measure contribution
                return_dict_in_generate=True
            )

        # Step 4: Extract generated tokens and attention weights from the model output
        generated_ids = outputs.sequences[0]
        generated_tokens = self.tokenizer_generator.decode(generated_ids, skip_special_tokens=True).split()

        # Extract attentions from the last layer (shape: [1, num_heads, target_len, source_len])
        attentions = outputs.decoder_attentions[-1].squeeze(0).mean(0).cpu().numpy()

        # We want to summarize attentions by summing over attention heads, and then normalize per token
        attention_matrix = attentions[:, :len(retrieved_texts)]  # Only consider attentions over retrieved documents

        # Normalize each row (for each token) to obtain posterior probabilities over documents
        normalized_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

        # Step 5: Plot the heatmap for posterior probabilities
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            normalized_matrix,
            cmap="Blues",
            xticklabels=[f"Doc {i+1}" for i in range(top_k)],
            yticklabels=["BOS"] + generated_tokens,
            cbar_kws={'label': 'Document Posterior Probability'}
        )
        plt.xlabel("Documents")
        plt.ylabel("Generated Tokens")
        plt.title("Document Posterior Probabilities for Each Generated Token")

        if writer:
            # Save the plot to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Convert buffer image to a TensorBoard-friendly format
            image = Image.open(buffer)
            transform = transforms.ToTensor()
            image_tensor = transform(image)

            # Log the image to TensorBoard
            writer.add_image("Posterior_Heatmap", image_tensor, global_step=100)
            self.log.info(f"Logged heatmap to TensorBoard")

        return normalized_matrix

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

    def train_scratch_rag_model(
            self,
            queries,
            target_outputs,
            learning_rate=1e-5,
            epochs=3,
            checkpoint_interval=1,
            resume_from_checkpoint=None,
            writer=None
        ):
        """Train the RAG model with checkpointing support."""
        if resume_from_checkpoint:
            self.log.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            self.log.info("Starting training.")

        # log all the parameters
        self.log.info(f"Learning rate: {learning_rate}")
        self.log.info(f"Epochs: {epochs}")
        self.log.info(f"Checkpoint interval: {checkpoint_interval}")
        self.log.info(f"Resume from checkpoint: {resume_from_checkpoint}")

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        if resume_from_checkpoint:
            self.load_model(resume_from_checkpoint)
            start_epoch = int(resume_from_checkpoint.split('_')[-1])

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()  # Record the start time of the epoch
            total_loss = 0
            self.log.info(f"Epoch {epoch + 1} started.")

            for i, (query, target_output) in enumerate(tqdm(zip(queries, target_outputs), total=len(queries), desc=f"Epoch {epoch + 1}")):
                optimizer.zero_grad()
                loss = self.calculate_loss(query, target_output, criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if i % 100 == 0 and i != 0:  # Log every 100 steps
                    self.log.info(f"Epoch {epoch + 1}, Step {i} : Loss = {loss.item():.4f}")

            average_loss = total_loss / len(queries)
            self.log.info(f"Epoch {epoch + 1} complete. Average Loss: {average_loss:.4f}")
            if writer:
                writer.add_scalar("Average Epoch Loss", average_loss, epoch + 1)

            # Record the time taken for this epoch
            epoch_duration = time.time() - epoch_start_time
            self.log.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")

            # Save the checkpoint at the end of each epoch or at a set interval
            if (epoch + 1) % checkpoint_interval == 0 and epoch != epochs - 1:
                self.save_model(
                    save_directory=os.path.join(self.log_dir, f"checkpoint_{epoch + 1}")
                )

    def inference(self, query):
        self.retriever.eval()
        self.generator.eval()

        retrieved_docs = self.retrieve_documents(query, top_k=self.top_k)

        return self.generate_output(query, retrieved_docs)

    # Training and evaluation using rag-mini-wikipedia dataset
    def pipeline(self,
                 num_train_epochs=3,
                 learning_rate=1e-5,
                 save_model=False,
                 checkpoint_interval=1,
                 resume_from_checkpoint=None,
        ):
        writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))  # TensorBoard writer

        # Call training function with the option to resume from a checkpoint
        self.log.info("Training RAG model with rag-mini-wikipedia dataset...")
        self.train_scratch_rag_model(
            self.train_queries, self.train_target_outputs,
            learning_rate=learning_rate,
            epochs=num_train_epochs,
            checkpoint_interval=checkpoint_interval,
            resume_from_checkpoint=resume_from_checkpoint,
            writer=writer
        )

        self.log.info("Saving the trained model...")
        if save_model:
            save_pth = os.path.join(self.log_dir, "rag_trained_model_wikiqa")
            self.save_model(save_directory=save_pth)

        self.log.info("Evaluating the model...")
        top_k_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.evaluate(top_k_values=top_k_values)

        writer.close()
        self.log.info("Pipeline finished.")

if __name__ == "__main__":
    log_dir = './logs'
    model = ScratchRAGModel(log_dir)
    model.pipeline(
        num_train_epochs = 3,
        learning_rate = 1e-5,
        save_model = True,
        load_dataset_ratios = 1.0
    )

    query = "What is the capital of France?"
    model.forward(query)