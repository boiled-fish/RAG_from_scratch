import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Get the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
import argparse
from rag.rag_scratch import ScratchRAGModel
from rag.rag_fine_tune import FineTuneRAGModel
from rag.Retriever import Retriever

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def main(args):
    # Define parameters for the pipeline
    num_train_epochs = 20
    learning_rate = 3e-3
    logging_dir = './log'
    save_model = True
    checkpoint_interval = 1

    if args.mode == "scratch":
        model = ScratchRAGModel(logging_dir)
        model.pipeline(
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        save_model = save_model,
        checkpoint_interval = checkpoint_interval,
        resume_from_checkpoint="./log/scratch_rag/checkpoint_11",
            load_dataset_ratio=1.0
        )
        query = "What did The Legal Tender Act of 1862 establish?"
        print(model.inference(query))
    elif args.mode == "fine_tune":
        model = FineTuneRAGModel(logging_dir)
        model.pipeline(
            num_train_epochs = num_train_epochs,
            learning_rate = learning_rate,
            save_model = save_model,
            checkpoint_interval = checkpoint_interval
        )
    elif args.mode == "retriever":
        retriever = Retriever(
            log_dir='./log',
            embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_dir='./hugging_face_models'
        )
        retriever.pipeline(learning_rate=1e-6, num_epochs=50000, resume=False)
        
        #query = "What is the capital of France?"
        #print(retriever.inference(query, top_k=5))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, default="scratch")
    args = args.parse_args()

    main(args)

