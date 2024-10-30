import os
import sys

# Get the project root directory dynamically
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to sys.path
sys.path.append(project_root)

import argparse

from rag.rag_scratch import ScratchRAGModel

def main(args):
    # Define parameters for the pipeline
    num_train_epochs = 50
    learning_rate = 3e-5
    logging_dir = './log'
    save_model = True
    checkpoint_interval = 10

    model = ScratchRAGModel(logging_dir)
    model.pipeline(
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        save_model = save_model,
        checkpoint_interval = checkpoint_interval
    )

    query = "What did The Legal Tender Act of 1862 establish?"
    print(model.forward(query))

def use_model(args):
    model = ScratchRAGModel('./log')
    model.load_model("./log/scratch_rag/rag_trained_model_wikiqa")

    query = "What did The Legal Tender Act of 1862 establish?"
    print(model.inference(query))

if __name__ == "__main__":
    # add argparse for different modes
    parser = argparse.ArgumentParser(description="Run the RAG model in different modes")
    parser.add_argument("--mode", choices=["scratch", "fine_tune", "use"], default="scratch", help="Mode: scratch, fine_tune, use")
    args = parser.parse_args()

    if args.mode == "scratch":
        main(args)
    elif args.mode == "use":
        use_model(args)

    

