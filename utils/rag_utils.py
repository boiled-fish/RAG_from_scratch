import os
import torch
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu

def save_model(model, save_directory="saved_model"):
    """Save the trained retriever and generator models."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save retriever (BERT)
    retriever_dir = os.path.join(save_directory, "retriever")
    model.retriever.save_pretrained(retriever_dir)
    model.tokenizer_retriever.save_pretrained(retriever_dir)

    # Save generator (BART)
    generator_dir = os.path.join(save_directory, "generator")
    model.generator.save_pretrained(generator_dir)
    model.tokenizer_generator.save_pretrained(generator_dir)
    print(f"Model saved to {save_directory}")


def evaluate_rag_model(model, test_queries, documents, target_outputs):
    """Evaluate the RAG model on a set of test queries."""
    model.eval()
    
    # Load evaluation metrics using the new 'evaluate' library
    rouge = load("rouge")
    bleu_scores = []

    generated_outputs = []
    for query, target_output in zip(test_queries, target_outputs):
        with torch.no_grad():
            generated_output = model(query, documents)
            generated_outputs.append(generated_output)

            # Calculate BLEU score for the generated output
            bleu_score = sentence_bleu([target_output.split()], generated_output.split())
            bleu_scores.append(bleu_score)

            # Add the results to ROUGE metric
            rouge.add(prediction=generated_output, reference=target_output)
    
    # Compute average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    # Compute ROUGE scores
    rouge_result = rouge.compute()

    print("Evaluation Results:")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")
    print(f"ROUGE Score: {rouge_result}")

    return generated_outputs, avg_bleu_score, rouge_result