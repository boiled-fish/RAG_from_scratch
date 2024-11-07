import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever, DPRContextEncoder, DPRContextEncoderTokenizerFast, AutoTokenizer, AutoModel
from evaluate import load
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from utils.custom_logger import CustomLogger

class FineTuneRAGModel(nn.Module):
    def __init__(self, log_dir='./log', model_name='facebook/rag-token-base', args=None, resume_pth=None, start_epoch=0):
        super(FineTuneRAGModel, self).__init__()
        # Initialize logger
        self.logger = CustomLogger(log_dir_base=log_dir, logger_name="fine_tune_rag")
        self.log_dir = self.logger.get_log_dir()
        self.log = self.logger.get_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"Using device: {self.device}")

        # Load the passages dataset
        self.log.info("Loading passages from rag-datasets/rag-mini-wikipedia...")
        passages_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'text-corpus', split='passages')

        self.start_epoch = start_epoch
        if resume_pth == None:
          # Load pre-trained RAG model and tokenizer
          self.tokenizer = RagTokenizer.from_pretrained(model_name, cache_dir='./hugging_face_models')
          self.model = RagTokenForGeneration.from_pretrained(model_name, cache_dir='./hugging_face_models')

          # Set up the retriever with the passages
          self.log.info("Setting up retriever...")
          self.retriever = RagRetriever.from_pretrained(
              model_name,
              index_name='exact',
              passages_dataset=passages_dataset,
              use_dummy_dataset=True,  # Avoid loading full dataset
              cache_dir='./hugging_face_models'
          )
        elif resume_pth != None:
          self.log.info("Resuming training from checkpoint.")
          self.load_model(resume_pth)

        # Set the retriever in the model
        self.model.set_retriever(self.retriever)

        # Move model to device
        self.model.to(self.device)

        # Load the QA dataset
        self.log.info("Loading question-answer dataset...")
        qa_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'question-answer', cache_dir='./datasets')

        # Split the data into training and evaluation sets
        train_test_split = qa_dataset['test'].train_test_split(test_size=0.2)
        self.train_dataset = train_test_split['train']
        self.eval_dataset = train_test_split['test']

        self.log.info("Initialization complete.")

    def train_model(self, epochs=30, batch_size=8, learning_rate=1e-5, checkpoint_interval=10, writer=None):
        """Train the RAG model."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log.info("Starting training.")

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_dataloader) * epochs

        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            self.log.info(f"Epoch {epoch + 1}/{epochs} started.")

            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                questions = batch['question']
                answers = batch['answer']

                # Tokenize the questions and answers
                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                labels = self.tokenizer(
                    answers,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).input_ids.to(self.device)

                # Replace padding token ids with -100 to ignore them during loss computation
                labels[labels == self.tokenizer.generator.pad_token_id] = -100

                optimizer.zero_grad()

                # **Pass attention_mask to the model**
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    labels=labels
                )

                # **Ensure loss is scalar by computing the mean**
                loss = outputs.loss.mean()  # Compute mean over batch

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if writer:
                    writer.add_scalar("Training Loss", loss.item(), epoch * len(train_dataloader) + i)

                if i % 100 == 0 and i != 0:
                    self.log.info(f"Epoch {epoch + 1}, Step {i}: Loss = {loss.item():.4f}")

            average_loss = total_loss / len(train_dataloader)
            self.log.info(f"Epoch {epoch + 1} complete. Average Loss: {average_loss:.4f}")
            if writer:
                writer.add_scalar("Average Epoch Loss", average_loss, epoch + 1)

            epoch_duration = time.time() - epoch_start_time
            self.log.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(os.path.join(self.log_dir, f"checkpoint_{epoch + 1}"))

    def evaluate_model1(self, batch_size=8, writer=None):
        """Evaluate the RAG model."""
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        total_bleu_score = 0.0
        total_rouge_score = 0.0
        total_exact_match = 0.0
        total_samples = 0

        rouge_metric = load("rouge")
        self.log.info("Starting evaluation.")

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                questions = batch['question']
                answers = batch['answer']

                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                # **Pass attention_mask to the generate function**
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
                generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for generated_answer, reference in zip(generated_answers, answers):
                    total_samples += 1

                    # Exact Match
                    if generated_answer.strip() == reference.strip():
                        total_exact_match += 1

                    # BLEU Score
                    bleu_score = sentence_bleu([reference.split()], generated_answer.split())
                    total_bleu_score += bleu_score

                    # ROUGE-L Score
                    rouge_result = rouge_metric.compute(
                        predictions=[generated_answer],
                        references=[reference],
                        rouge_types=["rougeL"]
                    )
                    total_rouge_score += rouge_result["rougeL"]

        avg_exact_match = (total_exact_match / total_samples) * 100
        avg_bleu_score = (total_bleu_score / total_samples) * 100
        avg_rouge_score = (total_rouge_score / total_samples) * 100

        self.log.info(f"Evaluation Results:")
        self.log.info(f"  Exact Match: {avg_exact_match:.2f}%")
        self.log.info(f"  Average BLEU Score: {avg_bleu_score:.2f}%")
        self.log.info(f"  Average ROUGE-L Score: {avg_rouge_score:.2f}%")

        if writer:
            writer.add_scalar("Evaluation/Exact_Match", avg_exact_match)
            writer.add_scalar("Evaluation/BLEU_Score", avg_bleu_score)
            writer.add_scalar("Evaluation/ROUGE_L_Score", avg_rouge_score)

    def evaluate_model(self, batch_size=8, writer=None):
        """Evaluate the RAG model."""
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        total_bleu_score = 0.0
        total_rouge_score = 0.0
        total_exact_match = 0.0
        total_samples = 0

        rouge_metric = load("rouge")
        self.log.info("Starting evaluation.")

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                questions = batch['question']
                answers = batch['answer']

                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                # **Pass attention_mask to the generate function**
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
                generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for generated_answer, reference in zip(generated_answers, answers):
                    total_samples += 1

                    # Exact Match
                    if generated_answer.strip() == reference.strip():
                        total_exact_match += 1

                    # BLEU Score
                    bleu_score = sentence_bleu([reference.split()], generated_answer.split())
                    total_bleu_score += bleu_score

                    # ROUGE-L Score
                    rouge_result = rouge_metric.compute(
                        predictions=[generated_answer],
                        references=[reference],
                        rouge_types=["rougeL"]
                    )
                    total_rouge_score += rouge_result["rougeL"]

        avg_exact_match = (total_exact_match / total_samples) * 100
        avg_bleu_score = (total_bleu_score / total_samples) * 100
        avg_rouge_score = (total_rouge_score / total_samples) * 100

        self.log.info(f"Evaluation Results:")
        self.log.info(f"  Exact Match: {avg_exact_match:.2f}%")
        self.log.info(f"  Average BLEU Score: {avg_bleu_score:.2f}%")
        self.log.info(f"  Average ROUGE-L Score: {avg_rouge_score:.2f}%")

        if writer:
            writer.add_scalar("Evaluation/Exact_Match", avg_exact_match)
            writer.add_scalar("Evaluation/BLEU_Score", avg_bleu_score)
            writer.add_scalar("Evaluation/ROUGE_L_Score", avg_rouge_score)


    def inference(self, query):
        """Generate an answer for a given query."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
            generated_ids = self.model.generate(input_ids=inputs['input_ids'])
            generated_answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_answer

    def save_model(self, save_directory):
        """Save the model and tokenizer."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_pth = os.path.join(save_directory, "model")
        tokenizer_pth = os.path.join(save_directory, "tokenizer")
        retriever_pth = os.path.join(save_directory, "retriever")

        self.model.save_pretrained(model_pth)
        self.tokenizer.save_pretrained(tokenizer_pth)
        self.retriever.save_pretrained(retriever_pth)

        self.log.info(f"Model saved to {save_directory}")

    def load_model(self, load_directory):
        """Load the model and tokenizer."""
        model_pth = os.path.join(load_directory, "model")
        tokenizer_pth = os.path.join(load_directory, "tokenizer")
        retriever_pth = os.path.join(load_directory, "retriever")
        # Define paths to the sub-tokenizers
        generator_tokenizer_path = os.path.join(tokenizer_pth, "question_encoder_tokenizer")
        question_encoder_tokenizer_path = os.path.join(tokenizer_pth, "generator_tokenizer")

        # Load the tokenizer
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(question_encoder_tokenizer_path)
        self.tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)
        self.log.info(f"Tokenizer loaded from {load_directory}")

        # Load the model
        self.model = RagTokenForGeneration.from_pretrained(model_pth)
        self.log.info(f"Model loaded from {load_directory}")

        # Load the retriever
        self.retriever = RagRetriever.from_pretrained(retriever_pth)
        self.log.info(f"Retriever loaded from {load_directory}")

    def pipeline(self,
                 num_train_epochs=3,
                 batch_size=8,
                 learning_rate=1e-5,
                 save_model=False,
                 checkpoint_interval=1):
        writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))

        self.train_model(
            epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_interval=checkpoint_interval,
            writer=writer
        )

        if save_model:
            save_path = os.path.join(self.log_dir, "fine_tuned_rag_model")
            self.save_model(save_directory=save_path)

        self.evaluate_model(batch_size=batch_size, writer=writer)

        writer.close()
        self.log.info("Pipeline finished.")
    def __init__(self, log_dir='./log', model_name='facebook/rag-token-base', args=None, resume_pth=None, start_epoch=0):
        super(FineTuneRAGModel, self).__init__()
        # Initialize logger
        self.logger = CustomLogger(log_dir_base=log_dir, logger_name="fine_tune_rag")
        self.log_dir = self.logger.get_log_dir()
        self.log = self.logger.get_logger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"Using device: {self.device}")

        # Load the passages dataset
        self.log.info("Loading passages from rag-datasets/rag-mini-wikipedia...")
        passages_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'text-corpus', split='passages')

        self.start_epoch = start_epoch
        if resume_pth == None:
          # Load pre-trained RAG model and tokenizer
          self.tokenizer = RagTokenizer.from_pretrained(model_name, cache_dir='./hugging_face_models')
          self.model = RagTokenForGeneration.from_pretrained(model_name, cache_dir='./hugging_face_models')

          # Set up the retriever with the passages
          self.log.info("Setting up retriever...")
          self.retriever = RagRetriever.from_pretrained(
              model_name,
              index_name='exact',
              passages_dataset=passages_dataset,
              use_dummy_dataset=True,  # Avoid loading full dataset
              cache_dir='./hugging_face_models'
          )
        elif resume_pth != None:
          self.log.info("Resuming training from checkpoint.")
          self.load_model(resume_pth)

        # Set the retriever in the model
        self.model.set_retriever(self.retriever)

        # Move model to device
        self.model.to(self.device)

        # Load the QA dataset
        self.log.info("Loading question-answer dataset...")
        qa_dataset = load_dataset('rag-datasets/rag-mini-wikipedia', 'question-answer', cache_dir='./datasets')

        # Split the data into training and evaluation sets
        train_test_split = qa_dataset['test'].train_test_split(test_size=0.2)
        self.train_dataset = train_test_split['train']
        self.eval_dataset = train_test_split['test']

        self.log.info("Initialization complete.")

    def train_model(self, epochs=30, batch_size=8, learning_rate=1e-5, checkpoint_interval=10, writer=None):
        """Train the RAG model."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log.info("Starting training.")

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_dataloader) * epochs

        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            self.log.info(f"Epoch {epoch + 1}/{epochs} started.")

            for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                questions = batch['question']
                answers = batch['answer']

                # Tokenize the questions and answers
                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                labels = self.tokenizer(
                    answers,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).input_ids.to(self.device)

                # Replace padding token ids with -100 to ignore them during loss computation
                labels[labels == self.tokenizer.generator.pad_token_id] = -100

                optimizer.zero_grad()

                # **Pass attention_mask to the model**
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    labels=labels
                )

                # **Ensure loss is scalar by computing the mean**
                loss = outputs.loss.mean()  # Compute mean over batch

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if writer:
                    writer.add_scalar("Training Loss", loss.item(), epoch * len(train_dataloader) + i)

                if i % 100 == 0 and i != 0:
                    self.log.info(f"Epoch {epoch + 1}, Step {i}: Loss = {loss.item():.4f}")

            average_loss = total_loss / len(train_dataloader)
            self.log.info(f"Epoch {epoch + 1} complete. Average Loss: {average_loss:.4f}")
            if writer:
                writer.add_scalar("Average Epoch Loss", average_loss, epoch + 1)

            epoch_duration = time.time() - epoch_start_time
            self.log.info(f"Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds")

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(os.path.join(self.log_dir, f"checkpoint_{epoch + 1}"))

    def evaluate_model1(self, batch_size=8, writer=None):
        """Evaluate the RAG model."""
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        total_bleu_score = 0.0
        total_rouge_score = 0.0
        total_exact_match = 0.0
        total_samples = 0

        rouge_metric = load("rouge")
        self.log.info("Starting evaluation.")

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                questions = batch['question']
                answers = batch['answer']

                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                # **Pass attention_mask to the generate function**
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
                generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for generated_answer, reference in zip(generated_answers, answers):
                    total_samples += 1

                    # Exact Match
                    if generated_answer.strip() == reference.strip():
                        total_exact_match += 1

                    # BLEU Score
                    bleu_score = sentence_bleu([reference.split()], generated_answer.split())
                    total_bleu_score += bleu_score

                    # ROUGE-L Score
                    rouge_result = rouge_metric.compute(
                        predictions=[generated_answer],
                        references=[reference],
                        rouge_types=["rougeL"]
                    )
                    total_rouge_score += rouge_result["rougeL"]

        avg_exact_match = (total_exact_match / total_samples) * 100
        avg_bleu_score = (total_bleu_score / total_samples) * 100
        avg_rouge_score = (total_rouge_score / total_samples) * 100

        self.log.info(f"Evaluation Results:")
        self.log.info(f"  Exact Match: {avg_exact_match:.2f}%")
        self.log.info(f"  Average BLEU Score: {avg_bleu_score:.2f}%")
        self.log.info(f"  Average ROUGE-L Score: {avg_rouge_score:.2f}%")

        if writer:
            writer.add_scalar("Evaluation/Exact_Match", avg_exact_match)
            writer.add_scalar("Evaluation/BLEU_Score", avg_bleu_score)
            writer.add_scalar("Evaluation/ROUGE_L_Score", avg_rouge_score)

    def evaluate_model(self, batch_size=8, writer=None):
        """Evaluate the RAG model."""
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size)
        total_bleu_score = 0.0
        total_rouge_score = 0.0
        total_exact_match = 0.0
        total_samples = 0

        rouge_metric = load("rouge")
        self.log.info("Starting evaluation.")

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                questions = batch['question']
                answers = batch['answer']

                inputs = self.tokenizer(
                    questions,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)

                # **Pass attention_mask to the generate function**
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Added attention_mask
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
                generated_answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for generated_answer, reference in zip(generated_answers, answers):
                    total_samples += 1

                    # Exact Match
                    if generated_answer.strip() == reference.strip():
                        total_exact_match += 1

                    # BLEU Score
                    bleu_score = sentence_bleu([reference.split()], generated_answer.split())
                    total_bleu_score += bleu_score

                    # ROUGE-L Score
                    rouge_result = rouge_metric.compute(
                        predictions=[generated_answer],
                        references=[reference],
                        rouge_types=["rougeL"]
                    )
                    total_rouge_score += rouge_result["rougeL"]

        avg_exact_match = (total_exact_match / total_samples) * 100
        avg_bleu_score = (total_bleu_score / total_samples) * 100
        avg_rouge_score = (total_rouge_score / total_samples) * 100

        self.log.info(f"Evaluation Results:")
        self.log.info(f"  Exact Match: {avg_exact_match:.2f}%")
        self.log.info(f"  Average BLEU Score: {avg_bleu_score:.2f}%")
        self.log.info(f"  Average ROUGE-L Score: {avg_rouge_score:.2f}%")

        if writer:
            writer.add_scalar("Evaluation/Exact_Match", avg_exact_match)
            writer.add_scalar("Evaluation/BLEU_Score", avg_bleu_score)
            writer.add_scalar("Evaluation/ROUGE_L_Score", avg_rouge_score)


    def inference(self, query):
        """Generate an answer for a given query."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
            generated_ids = self.model.generate(input_ids=inputs['input_ids'])
            generated_answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_answer

    def save_model(self, save_directory):
        """Save the model and tokenizer."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_pth = os.path.join(save_directory, "model")
        tokenizer_pth = os.path.join(save_directory, "tokenizer")
        retriever_pth = os.path.join(save_directory, "retriever")

        self.model.save_pretrained(model_pth)
        self.tokenizer.save_pretrained(tokenizer_pth)
        self.retriever.save_pretrained(retriever_pth)

        self.log.info(f"Model saved to {save_directory}")

    def load_model(self, load_directory):
        """Load the model and tokenizer."""
        model_pth = os.path.join(load_directory, "model")
        tokenizer_pth = os.path.join(load_directory, "tokenizer")
        retriever_pth = os.path.join(load_directory, "retriever")
        # Define paths to the sub-tokenizers
        generator_tokenizer_path = os.path.join(tokenizer_pth, "question_encoder_tokenizer")
        question_encoder_tokenizer_path = os.path.join(tokenizer_pth, "generator_tokenizer")

        # Load the tokenizer
        generator_tokenizer = AutoTokenizer.from_pretrained(generator_tokenizer_path)
        question_encoder_tokenizer = AutoTokenizer.from_pretrained(question_encoder_tokenizer_path)
        self.tokenizer = RagTokenizer(question_encoder_tokenizer, generator_tokenizer)
        self.log.info(f"Tokenizer loaded from {load_directory}")

        # Load the model
        self.model = RagTokenForGeneration.from_pretrained(model_pth)
        self.log.info(f"Model loaded from {load_directory}")

        # Load the retriever
        self.retriever = RagRetriever.from_pretrained(retriever_pth)
        self.log.info(f"Retriever loaded from {load_directory}")

    def pipeline(self,
                 num_train_epochs=3,
                 batch_size=8,
                 learning_rate=1e-5,
                 save_model=False,
                 checkpoint_interval=1):
        writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))

        self.train_model(
            epochs=num_train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_interval=checkpoint_interval,
            writer=writer
        )

        if save_model:
            save_path = os.path.join(self.log_dir, "fine_tuned_rag_model")
            self.save_model(save_directory=save_path)

        self.evaluate_model(batch_size=batch_size, writer=writer)

        writer.close()
        self.log.info("Pipeline finished.")