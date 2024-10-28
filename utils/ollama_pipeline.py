import argparse
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from transformers import BertTokenizer, BertModel
import faiss
import numpy as np
import clip  # Ensure CLIP is installed
import json

# TODO: add image analyser class, for each image, get a description of the image, not only the embedding

# 1. Web Scraper Class
class WebScraper:
    def __init__(self, url):
        self.url = url

    def scrape(self):
        """
        Scrapes text and images from the given webpage URL.
        Returns text content and a list of image URLs.
        """
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all text from the webpage
        text = soup.get_text(separator=' ', strip=True)

        # Extract all image URLs from the webpage
        images = []
        for img_tag in soup.find_all('img'):
            img_url = img_tag.get('src')
            if img_url and img_url.startswith('http'):
                images.append(img_url)
        
        return text, images

# 2. Text Embedding Class (Using BERT)
class TextEmbedder:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()

    def create_embedding(self, text):
        """
        Generates a text embedding using a pre-trained BERT model and ensures the embedding is 2D (n_samples, embedding_dim).
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pool over the sequence dimension
            
        embedding = embedding.squeeze()  # Remove any extra dimensions (in case it's 3D, e.g., (1, 768))
        
        if len(embedding.shape) == 1:  # Ensure it's 2D (1, embedding_dim)
            embedding = embedding.unsqueeze(0)

        print(f"Text Embedding Shape: {embedding.shape}")  # Debugging the shape
        return embedding.cpu().numpy()  # Return as NumPy array

# 3. Image Embedding Class (for CLIP or ViT)
class ImageEmbedder:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    def create_embedding(self, image_url):
        """
        Generates an image embedding using a pre-trained CLIP model and ensures the embedding is 2D (n_samples, embedding_dim).
        """
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

        embedding = embedding.squeeze()  # Remove any extra dimensions (in case it's 3D, e.g., (1, 512))
        
        if len(embedding.shape) == 1:  # Ensure it's 2D (1, embedding_dim)
            embedding = embedding.unsqueeze(0)

        print(f"Image Embedding Shape: {embedding.shape}")  # Debugging the shape
        return embedding.cpu().numpy()  # Return as NumPy array
    
def normalize_vector(vector):
    """
    Normalizes a vector to have a unit norm (for cosine similarity).
    """
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / norm

# 4. FAISS Vector Store Class
class VectorStore:
    def __init__(self, dimension):
        """
        Initializes FAISS with an inner product index for cosine similarity.
        """
        self.index = faiss.IndexFlatIP(dimension)  # Use IndexFlatIP for inner product (cosine similarity)
        self.dimension = dimension

    def add_embeddings(self, embeddings):
        """
        Adds the given embeddings to the FAISS index.
        """
        embeddings = np.array(embeddings)

        # Check that the dimensionality of the embeddings matches the expected dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimensionality {embeddings.shape[1]} does not match expected {self.dimension}")

        print(f"Embedding shape before adding to FAISS: {embeddings.shape}")  # Debugging the shape
        self.index.add(embeddings.astype('float32'))  # Ensure the embeddings are float32

class LlamaModel:
    def __init__(self, api_url="http://localhost:11434/api/generate"):
        self.api_url = api_url

    def query(self, user_input, text_results, image_results):
        """
        Sends a prompt to the Llama model API and returns a structured response based on the user input and relevant documents.
        """
        prompt = """
        You are an intelligent assistant that provides structured and concise answers to knowledge-based questions. 
        Below is a structured summary of the most relevant information retrieved from various documents, followed by the user's question.

        User's question:
        {user_input}

        Relevant information:
        Text results:
        - {text_results}

        Image results:
        - {image_results}

        Based on the relevant information, answer the user's question clearly and accurately. Your response should be structured into the following sections:
        1. **Query**: Restate the user's query.
        2. **Information from Text**: Summarize the relevant information from text results.
        3. **Information from Images**: Summarize the relevant information from image results (if applicable).
        4. **Conclusion**: Provide a concise response based on the retrieved information.
        If the information is insufficient or unrelated, let the user know.
        """
        
        # Prepare data for the API request
        data = {
            "model": "llama3.1:8b",
            "prompt": prompt.format(user_input=user_input, text_results=text_results, image_results=image_results)
        }

        headers = {'Content-Type': 'application/json'}

        try:
            # Send request to the Llama model API
            response = requests.post(self.api_url, data=json.dumps(data), headers=headers, stream=True)

            full_response = []
            for line in response.iter_lines():
                if line:
                    # Parse each line of response
                    decode_line = json.loads(line.decode('utf-8'))
                    full_response.append(decode_line['response'])

            # Join all response parts and return the final response
            return "".join(full_response)

        finally:
            response.close()

# 6. Main Pipeline Class
class Pipeline:
    def __init__(self, url, device='cuda' if torch.cuda.is_available() else 'cpu', k=5):
        self.url = url
        self.device = device
        self.k = k

        # Initialize components
        self.scraper = WebScraper(url)
        
        # Define the correct dimensions for text and image embeddings
        self.text_vector_store = VectorStore(dimension=768)  # BERT embedding dimension
        self.image_vector_store = VectorStore(dimension=512)  # CLIP embedding dimension
        
        self.text_embedder = TextEmbedder(device)
        self.image_embedder = ImageEmbedder(device)
        self.llama_model = LlamaModel()

    def combine_embeddings(self, text_embedding=None, image_embedding=None):
        """
        Combines text and image embeddings by concatenation.
        If only one embedding is provided, returns that embedding.
        """
        if text_embedding is not None and image_embedding is not None:
            return np.concatenate((text_embedding, image_embedding), axis=1)
        elif text_embedding is not None:
            return text_embedding
        elif image_embedding is not None:
            return image_embedding
        else:
            raise ValueError("Both text and image embeddings cannot be None.")
    
    def run(self, query_text=None, query_image_url=None):
        # Step 1: Scrape the webpage and store the document content
        webpage_text, image_urls = self.scraper.scrape()
        print(f"Scraped {len(image_urls)} images and extracted text from {self.url}")

        def chunk_text(text, chunk_size=200):
            """
            Breaks a large text into smaller chunks of a specified size.
            """
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        # Step 2: Break the webpage text into smaller chunks for indexing
        text_chunks = chunk_text(webpage_text, chunk_size=500)  # Adjust chunk_size as needed

        # Store the text chunks for retrieval later
        self.text_documents = text_chunks  # Store the chunks instead of the whole document

        # Step 3: Generate and store normalized text embeddings for each chunk
        for chunk in text_chunks:
            text_embedding = self.text_embedder.create_embedding(chunk)
            normalized_embedding = normalize_vector(text_embedding)  # Normalize the embedding
            self.text_vector_store.add_embeddings(normalized_embedding)

        # Step 4: Generate and store normalized image embeddings
        for image_url in image_urls:
            image_embedding = self.image_embedder.create_embedding(image_url)
            normalized_embedding = normalize_vector(image_embedding)  # Normalize the embedding
            self.image_vector_store.add_embeddings(normalized_embedding)

        # Step 5: Handle the query
        if query_text or query_image_url:
            self.handle_query(query_text=query_text, query_image_url=query_image_url)


    def handle_query(self, query_text=None, query_image_url=None):
        """
        Handles querying the vector store using either text or image queries.
        Retrieves the top-k most similar embeddings from the FAISS index and maps the results to the actual content.
        """
        text_results, image_results = [], []

        if query_text:
            query_text_embedding = self.text_embedder.create_embedding(query_text)
            normalized_query_embedding = normalize_vector(query_text_embedding)  # Normalize the query

            # Search the FAISS index for similar text embeddings
            print("Searching text vector store...")
            D, I = self.text_vector_store.index.search(normalized_query_embedding.astype('float32'), self.k)

            # Retrieve the actual text documents based on FAISS indices
            text_results = [self.text_documents[i] for i in I[0]]
            print(f"Top {self.k} most similar text results:")
            for idx, result in enumerate(text_results):
                print(f"{idx + 1}. {result[:200]}")  # Limit to 200 characters for readability

        if query_image_url:
            query_image_embedding = self.image_embedder.create_embedding(query_image_url)
            normalized_query_embedding = normalize_vector(query_image_embedding)  # Normalize the query

            # Search the FAISS index for similar image embeddings
            print("Searching image vector store...")
            D, I = self.image_vector_store.index.search(normalized_query_embedding.astype('float32'), self.k)

            # Retrieve the actual image URLs based on FAISS indices
            image_results = [self.image_documents[i] for i in I[0]]
            print(f"Top {self.k} most similar image results:")
            for idx, result in enumerate(image_results):
                print(f"{idx + 1}. {result}")  # Display image URLs

        # Generate structured response based on retrieved results
        self.generate_response_to_query(query_text=query_text, text_results=text_results, image_results=image_results)


    def generate_response_to_query(self, query_text=None, text_results=[], image_results=[]):
        """
        Generates a structured response to the query using the results retrieved from FAISS.
        """
        relevant_text_documents = '\n- '.join(text_results) if len(text_results) > 0 else 'No relevant text results found.'
        relevant_image_documents = '\n- '.join(image_results) if len(image_results) > 0 else 'No relevant image results found.'

        # Generate the response using the Llama model
        response = self.llama_model.query(
            user_input=query_text, 
            text_results=relevant_text_documents, 
            image_results=relevant_image_documents
        )
        print(f"Generated Response:\n{response}")


# 7. Command Line Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run a pipeline that scrapes text and images, stores embeddings in FAISS, and queries using text or image.")
    
    parser.add_argument('--url', type=str, required=True, help="URL of the webpage to scrape.")
    parser.add_argument('--query_text', type=str, help="Text query to search in the vector store.")
    parser.add_argument('--query_image_url', type=str, help="Image URL query to search in the vector store.")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top results to return from FAISS query.")
    
    return parser.parse_args()

# 8. Execute Pipeline from Command Line
if __name__ == "__main__":
    args = parse_args()

    # Initialize pipeline
    pipeline = Pipeline(url=args.url, k=args.top_k)

    # Run the pipeline with provided queries
    pipeline.run(query_text=args.query_text, query_image_url=args.query_image_url)
