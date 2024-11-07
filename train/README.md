# Training a RAG Model

This guide provides instructions for training a Retrieval-Augmented Generation (RAG) model. You can choose from the following training modes:
- Fine-tune both the retriever and generator using pre-trained models.
- Train both the retriever and generator from scratch.
- Train only the retriever component.
- Utilize LangChain to construct the RAG system.

## System Requirements

- Python version 3.9.19
- Ollama installed locally or accessible remotely
- Recommended: 32GB+ RAM for handling large document sets
- Recommended: 16GB+ GPU memory for efficient training
- Recommended: CUDA 12.5 GPU for optimal performance
- Alternatively, use Google Colab with GPU support (Jupyter notebooks available in `repo/train/colab`)

## Installation Steps

1. **Set up Python Environment:**
   ```bash
   conda create -n train_rag python=3.9.19
   conda activate train_rag
   ```

2. **Install Required Python Packages:**
   ```bash
   cd src/server
   pip install -r requirements.txt
   ```

3. **Configure Ollama Models:**
   ```bash
   ollama pull ollama/llama3.1:8b
   ollama pull ollama/nomic-embed-text-v3.small
   ```

## Training Process

[Optional] Initialize the Ollama Server if using LangChain or Ollama.

1. **Execute the Training Script:**
   ```bash
   python train.py --mode [joint_train, joint_train_from_scratch, retriever_only]
   ```
   The script will automatically create directories for datasets, pre-trained models, and logs. Model checkpoints will be saved in the log directory.

2. **Monitor Training Progress:**
   ```bash
   tensorboard --logdir log/[tensorboard_log_dir]
   ```

## Overview of Training Modes

 // Start of Selection
- **Fine-Tuning:** The `rag_fine_tune.py` script introduces the `FineTuneRAGModel` class, specifically crafted to fine-tune the Retrieval-Augmented Generation (RAG) model. It leverages meticulously curated datasets and robust evaluation methodologies to enhance model performance. This class manages the fine-tuning pipeline by initializing essential components such as `RagTokenizer`, `RagTokenForGeneration`, and `RagRetriever` from the Hugging Face Transformers library, utilizing the `facebook/rag-token-base` as the foundational model. The model employs the `rag-mini-wikipedia` dataset—a carefully selected subset of Wikipedia articles optimized for retrieval-augmented generation tasks—alongside dedicated question-answer datasets for comprehensive training and evaluation.

  Training is conducted using PyTorch, with the Adam optimizer responsible for updating model parameters. The class encompasses extensive training and testing functions, complemented by detailed logging through a custom logger and integrated TensorBoard support for real-time monitoring of training metrics. Upon completion, the `FineTuneRAGModel` produces a fully trained model capable of performing inference via its dedicated method, which generates accurate responses to input queries. Additionally, the class includes functionalities for saving and loading model checkpoints, ensuring seamless continuation of training sessions and facilitating deployment. Model performance is rigorously evaluated using BLEU and ROUGE metrics, providing quantitative assessments of the quality and relevance of the generated responses.

- **Scratch Training:** Implements the joint training of both the retriever and generator components using `bert-base-uncased` as the retriever model and `facebook/bart-large` as the generator. This approach utilizes the `rag-mini-wikipedia` dataset, a curated subset of Wikipedia articles optimized for retrieval-augmented generation tasks. The dataset is split into training and testing subsets, with approximately 80% allocated for training and 20% for testing to ensure robust evaluation. The retriever encodes queries and documents with BERT, storing document embeddings in a Faiss `IndexFlatL2` for efficient similarity search. Document embedding involves tokenizing passages with `BertTokenizer`, processing them through `BertModel`, and aggregating the embeddings. The generator employs `BartForConditionalGeneration` to produce responses based on the retrieved documents.

  Training is managed with PyTorch's Adam optimizer and `CrossEntropyLoss`, ensuring all parameters are trainable. The system supports checkpointing for saving and resuming training and utilizes a custom `CustomLogger` for detailed logging. Performance is evaluated using BLEU and ROUGE metrics, which assess the quality and relevance of the generated responses against the ground truth.

- **Retriever-Only Training:** The `rag_retriever.py` script focuses on training the retriever component of the Retrieval-Augmented Generation (RAG) model. It utilizes the `sentence-transformers/all-MiniLM-L6-v2` model to encode queries and documents into dense vector representations. These embeddings are then indexed using Faiss's `IndexFlatIP` to enable efficient similarity-based retrieval. The training process leverages the WikiQA dataset, divided into training and testing subsets to optimize and evaluate the retriever's performance. PyTorch's Adam optimizer is employed alongside `CrossEntropyLoss` to update the model parameters, aiming to maximize the similarity between relevant query-document pairs. The training incorporates in-batch negatives to enhance model effectiveness.

  Comprehensive logging is managed through the `CustomLogger`, and TensorBoard integration provides real-time monitoring of training metrics such as loss and accuracy. The script supports model checkpointing, allowing for the saving and loading of trained models for future inference tasks. Additionally, the pipeline includes encoding the entire corpus, building the FAISS index, and performing inference to retrieve relevant documents based on input queries.

# Datasets

- `rag-mini-wikipedia`(https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia): A curated subset of Wikipedia articles optimized for retrieval-augmented generation tasks.
- `WikiQA`(https://huggingface.co/datasets/microsoft/wiki_qa): A dataset containing question-answer pairs extracted from Wikipedia articles, used for training and evaluating the retriever.