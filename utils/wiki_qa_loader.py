import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Define the Dataset class for WikiQA
class WikiQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]  # Assuming Hugging Face Datasets format
        query = row['question']
        answer = row['answer']
        # Since 'document_title' might be relevant in context, include it
        passage = row['document_title'] + " " + answer

        return query, answer, passage

def preprocess_wikiqa():
    """
    Preprocess WikiQA dataset using the Hugging Face 'datasets' library.
    """
    # Load the train and validation splits
    train_dataset = load_dataset('wiki_qa', split='train')
    test_dataset = load_dataset('wiki_qa', split='validation')

    return train_dataset, test_dataset

def load_wikiqa_data(tokenizer, batch_size=8):
    """
    Load and preprocess the WikiQA dataset into DataLoader objects for training and testing.
    """
    train_dataset, test_dataset = preprocess_wikiqa()
    
    # Create dataset objects
    train_dataset = WikiQADataset(train_dataset, tokenizer)
    test_dataset = WikiQADataset(test_dataset, tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

