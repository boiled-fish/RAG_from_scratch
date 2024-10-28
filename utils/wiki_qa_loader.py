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
        row = self.data.iloc[index]
        query = row['Question']
        answer = row['Answer']
        passage = row['Passage']

        return query, answer, passage

def preprocess_wikiqa(file_path):
    """
    Preprocess WikiQA dataset into a format suitable for training the retriever.
    The dataset is assumed to be in a CSV format with 'Question', 'Answer', and 'Passage' columns.
    """
    df = pd.read_csv(file_path, delimiter='\t')
    
    # Filter positive and negative samples (assuming 1 = positive, 0 = negative)
    positive_samples = df[df['Label'] == 1]
    negative_samples = df[df['Label'] == 0]
    
    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df

def load_wikiqa_data(file_path, tokenizer, batch_size=8):
    """
    Load and preprocess the WikiQA dataset into DataLoader objects for training and testing.
    """
    train_df, test_df = preprocess_wikiqa(file_path)
    
    # Create dataset objects
    train_dataset = WikiQADataset(train_df, tokenizer)
    test_dataset = WikiQADataset(test_df, tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
