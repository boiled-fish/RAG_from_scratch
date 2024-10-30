import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.custom_logger import CustomLogger
from utils.rag_utils import save_model, evaluate_rag_model, load_checkpoint, save_checkpoint

class WikiQADataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            item['question'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                item['answer'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class FineTuneRAGModel:
    def __init__(
        self,
        model_name: str = "facebook/rag-sequence-nq",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Initialize the CustomLogger
        self.logger = CustomLogger(log_dir_base="./log/finetune_rag_logs", logger_name="finetune_rag")
        self.log_dir = self.logger.get_log_dir()
        self.log = self.logger.get_logger()

        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.retriever = None
        
        # Create cache directories
        os.makedirs("./hugging_face_models", exist_ok=True)
        os.makedirs("./datasets", exist_ok=True)
        
        self.log.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the pre-trained RAG model and tokenizer"""
        self.log.info(f"Loading model {self.model_name}...")
        
        # Initialize tokenizer
        self.tokenizer = RagTokenizer.from_pretrained(
            self.model_name,
            cache_dir='./hugging_face_models'
        )
        
        # Initialize retriever
        self.retriever = RagRetriever.from_pretrained(
            self.model_name,
            index_name="exact",
            use_dummy_dataset=False,
            cache_dir='./hugging_face_models'
        )
        
        # Initialize model
        self.model = RagSequenceForGeneration.from_pretrained(
            self.model_name,
            retriever=self.retriever,
            cache_dir='./hugging_face_models'
        ).to(self.device)
        
        self.log.info("Model loaded successfully!")

    def prepare_wikiqa_dataset(self, max_samples=None):
        """Load and prepare the WikiQA dataset"""
        self.log.info("Loading WikiQA dataset...")
        dataset = load_dataset('wiki_qa', cache_dir='./datasets')
        
        # Select subset of data if specified
        if max_samples:
            train_data = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
            val_data = dataset['validation'].select(range(min(max_samples//10, len(dataset['validation']))))
        else:
            train_data = dataset['train']
            val_data = dataset['validation']
            
        self.log.info(f"Training data size: {len(train_data)}")
        self.log.info(f"Validation data size: {len(val_data)}")
        
        # Create datasets
        train_dataset = WikiQADataset(self.tokenizer, train_data)
        val_dataset = WikiQADataset(self.tokenizer, val_data)
        
        return train_dataset, val_dataset

    def fine_tune(
        self,
        train_dataset,
        val_dataset,
        output_dir: str = "./rag_fine_tuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        resume_from_checkpoint: bool = False
    ):
        """Fine-tune the RAG model on WikiQA dataset"""
        self.log.info("Starting fine-tuning process...")
        
        # Setup TensorBoard
        writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.log_dir, 'logs'),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Load checkpoint if resuming
        if resume_from_checkpoint and os.path.exists(output_dir):
            self.log.info("Resuming from checkpoint...")
            checkpoint = load_checkpoint(self.model, None, output_dir)
            start_epoch = checkpoint.get('epoch', 0)
        else:
            start_epoch = 0

        # Training loop
        self.log.info(f"Starting training from epoch {start_epoch + 1}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the fine-tuned model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.log.info(f"Model fine-tuned and saved to {output_dir}")
        
        writer.close()

    def generate(
        self,
        question: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        num_beams: int = 4
    ):
        """Generate answer for a given question"""
        self.log.debug(f"Generating answer for question: {question}")
        
        inputs = self.tokenizer(
            question,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(self.device)

        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )

        return [
            self.tokenizer.decode(g, skip_special_tokens=True)
            for g in generated
        ]

    def pipeline(self, max_samples=1000):
        """Run the complete fine-tuning pipeline"""
        self.log.info("Starting RAG fine-tuning pipeline...")
        
        # Load pre-trained model
        self.load_model()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_wikiqa_dataset(max_samples=max_samples)
        
        # Fine-tune the model
        self.fine_tune(
            train_dataset,
            val_dataset,
            output_dir="./rag_fine_tuned_wikiqa",
            num_epochs=3,
            batch_size=4,
            resume_from_checkpoint=True
        )
        
        # Save the model
        save_model(self.model, save_directory="./rag_fine_tuned_wikiqa")
        
        # Evaluate the model
        self.log.info("Evaluating model...")
        sample_questions = [entry['question'] for entry in train_dataset.data[:5]]
        for question in sample_questions:
            answer = self.generate(question)[0]
            self.log.info(f"Q: {question}")
            self.log.info(f"A: {answer}\n")
        
        self.log.info("Pipeline completed successfully!")
        return self

# Example usage:
if __name__ == "__main__":
    model = FineTuneRAGModel()
    trained_model = model.pipeline(max_samples=1000)