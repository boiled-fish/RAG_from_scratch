import os
import sys
import time  # Added import for time

# Get the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
import faiss
import numpy as np
from datasets import load_dataset
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BartTokenizer
)
from torch.utils.tensorboard import SummaryWriter
import logging
from utils.custom_logger import CustomLogger

# Set environment variables for custom download paths
os.environ['HF_DATASETS_CACHE'] = './datasets'
os.environ['TRANSFORMERS_CACHE'] = './3rdPartyModels'

# Initialize the CustomLogger
logger = CustomLogger(log_dir_base="../log/fine_tune_rag_logs")
log_dir = logger.get_log_dir()

# Use a separate logger to avoid overwriting the logging module
log = logger.get_logger()

class RAGModel:
    def __init__(self, dataset_name='microsoft/wiki_qa', model_name='facebook/rag-sequence-nq'):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.context_encoder = None
        self.question_encoder = None
        self.index = None
        self.rag_retriever = None
        self.rag_model = None
        self.rag_tokenizer = None

    def load_dataset(self, split='train'):
        # Load the dataset
        self.dataset = load_dataset(self.dataset_name, split=split)

    def preprocess_data(self):
        # Define function to extract documents for the retriever
        def extract_corpus(example):
            # Combine the document title and answer for each entry as the "document"
            return {'text': example['document_title'] + " " + example['answer']}

        # Define function to extract question-answer pairs for the generator
        def extract_qa_pairs(example):
            # Use only positively labeled (correct) answers as QA pairs
            return {
                'question': example['question'],
                'answer': example['answer'] if example.get('label', 0) == 1 else ''  # Safeguard with .get()
            }

        # Process the corpus by mapping the extract_corpus function and filtering out empty documents
        self.corpus = self.dataset.map(
            extract_corpus, 
            remove_columns=self.dataset.column_names
        ).filter(lambda x: x['text'].strip() != '')  # Added .strip() to ensure no whitespace-only texts

        # Process QA pairs by mapping the extract_qa_pairs function and filtering out empty answers
        self.qa_pairs = self.dataset.map(
            extract_qa_pairs, 
            remove_columns=self.dataset.column_names
        ).filter(lambda x: x['answer'].strip() != '')  # Added .strip() for consistency

    def setup_retrieval(self):
        # Initialize the question and context encoders
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base", 
            cache_dir='../3rdPartyModels'
        ).to(self.device)
        
        self.context_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base", 
            cache_dir='../3rdPartyModels'
        ).to(self.device)
        
        # Initialize the tokenizers (no need to call .to(self.device) on tokenizers)
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base", 
            cache_dir='../3rdPartyModels'
        )
        
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base", 
            cache_dir='../3rdPartyModels'
        )

        # Encode the corpus using context encoder
        log.info("Encoding the corpus using the context encoder...")
        context_embeddings = []
        batch_size = 64  # Adjust batch size as per GPU memory
        for i in range(0, len(self.corpus), batch_size):
            batch_texts = self.corpus['text'][i:i+batch_size]
            inputs = self.context_tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512  # Adjust max_length as needed
            ).to(self.device)
            with torch.no_grad():
                embeddings = self.context_encoder(**inputs).pooler_output.cpu().numpy()
            context_embeddings.append(embeddings)
            log.info(f"Encoded batch {i//batch_size + 1}/{(len(self.corpus)+batch_size-1)//batch_size}")

        # Convert context embeddings list to a numpy array
        context_embeddings = np.vstack(context_embeddings)

        # Create a FAISS index for retrieval
        embedding_dim = context_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product index (cosine similarity)
        self.index.add(context_embeddings)
        log.info(f"FAISS index created with {self.index.ntotal} vectors.")

        # Load the dataset separately with `trust_remote_code=True` and specify a configuration
        wiki_dpr_dataset = load_dataset(
            "wiki_dpr", 
            "psgs_w100.nq.exact",  # Choose an appropriate config from the available ones
            split="train", 
            trust_remote_code=True, 
            cache_dir='../3rdPartyModels'
        )
        log.info("Loaded 'wiki_dpr' dataset for the retriever.")

        # Initialize the RagRetriever without passing the FAISS index directly
        self.rag_retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-base",
            use_dummy_dataset=False,  # Set to False since we're providing a passages_dataset
            cache_dir='../3rdPartyModels',
            passages_dataset=wiki_dpr_dataset  # Explicitly passing the dataset
        )
        log.info("RagRetriever initialized.")

        # Optionally, if you need to use the custom FAISS index, you would need to integrate it properly.
        # This might involve subclassing RagRetriever or modifying its internal components.
        # For simplicity, we're using the default retriever here.

    def initialize_model(self):
        # Use the RAG tokenizer
        self.rag_tokenizer = RagTokenizer.from_pretrained(
            "facebook/rag-token-base", 
            cache_dir='../3rdPartyModels',
            generator_tokenizer=BartTokenizer.from_pretrained("facebook/bart-large", cache_dir='../3rdPartyModels'),
            question_encoder_tokenizer=DPRQuestionEncoderTokenizer.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base", 
                cache_dir='../3rdPartyModels'
            )
        )
        
        # Initialize the RAG model using the retrieved configuration and models
        self.rag_model = RagSequenceForGeneration.from_pretrained(
            self.model_name, 
            cache_dir='../3rdPartyModels'
        ).to(self.device)
        log.info("RAG model initialized and moved to device.")

    def fine_tune(self, num_train_epochs=1, batch_size=2, learning_rate=3e-5, logging_dir='./logs'):
        # Initialize TensorBoard SummaryWriter
        writer = SummaryWriter(logging_dir)

        # Log the start of fine-tuning
        log.info("Starting fine-tuning process")

        # Preprocess the QA pairs for training
        def preprocess_function(examples):
            inputs = self.rag_tokenizer(
                examples['question'], 
                padding="max_length", 
                truncation=True, 
                max_length=128
            )
            labels = self.rag_tokenizer(
                examples['answer'], 
                padding="max_length", 
                truncation=True, 
                max_length=128
            )["input_ids"]
            inputs["labels"] = labels
            return inputs

        # Log preprocessing step
        log.info("Preprocessing QA pairs")
        processed_qa_pairs = self.qa_pairs.map(
            preprocess_function, 
            batched=True, 
            remove_columns=self.qa_pairs.column_names
        )
        log.info("Preprocessing completed.")

        # Prepare training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./rag-fine-tuned",
            evaluation_strategy="steps",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir=logging_dir,  # TensorBoard logging directory
            logging_steps=10,
            save_steps=10,
            eval_steps=10,
            logging_strategy="steps",
            save_total_limit=2,  # To limit the number of saved checkpoints
            remove_unused_columns=False,  # Important for certain models
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        )
        log.info("Training arguments set.")

        # Log model initialization
        log.info("Initializing the trainer")

        # Define a custom Trainer with logging hooks
        class CustomTrainer(Seq2SeqTrainer):
            def on_log(self, logs=None, **kwargs):
                # Log the metrics to TensorBoard
                if logs:
                    for key, value in logs.items():
                        writer.add_scalar(key, value, self.state.global_step)

                    # Log metrics to console
                    log.info(f"Step {self.state.global_step}: {logs}")

                super().on_log(logs, **kwargs)

            def training_step(self, model, inputs):
                # Custom training step to add batch-level logging
                start_time = time.time()
                loss = super().training_step(model, inputs)

                # Calculate time per batch and log it
                end_time = time.time()
                batch_time = end_time - start_time
                log.info(f"Processed batch in {batch_time:.2f} seconds - Loss: {loss.item()}")

                return loss

        # Create the trainer with custom logging
        trainer = CustomTrainer(
            model=self.rag_model,
            args=training_args,
            train_dataset=processed_qa_pairs,
            eval_dataset=processed_qa_pairs,  # Ideally, use a separate eval dataset
            tokenizer=self.rag_tokenizer,
            retriever=self.rag_retriever,  # Pass the retriever to the trainer
        )
        log.info("Trainer initialized.")

        # Log training start
        log.info("Starting training")

        # Train the model
        trainer.train()
        log.info("Training completed.")

        # Close the TensorBoard writer
        writer.close()
        log.info("TensorBoard writer closed.")

        # Log completion of fine-tuning
        log.info("Fine-tuning process completed")

    def inference(self, input_question):
        # Tokenize the input question
        inputs = self.rag_tokenizer(
            input_question, 
            return_tensors="pt"
        ).to(self.device)

        # Generate the answer using the RAG model
        generated_ids = self.rag_model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            num_beams=4,  # Adjust beam size as needed
            max_length=128,  # Adjust max length as needed
            early_stopping=True
        )

        # Decode the generated answer
        answer = self.rag_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answer

if __name__ == "__main__":  
    log.info("Instantiating RAGModel class")
    rag_model = RAGModel()

    log.info("Loading and preprocessing dataset")
    rag_model.load_dataset(split='train')
    rag_model.preprocess_data()

    log.info("Setting up retrieval")
    rag_model.setup_retrieval()

    log.info("Initializing RAG model")
    rag_model.initialize_model()

    log.info("Fine-tuning the RAG model")
    rag_model.fine_tune(num_train_epochs=1, batch_size=2, learning_rate=3e-5)

    log.info("Performing inference")
    input_question = "Who wrote the book 'Origin of Species'?"
    answer = rag_model.inference(input_question)
    print("Answer:", answer)