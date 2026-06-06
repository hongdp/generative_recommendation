"""Module for extracting movie title text embeddings using Sentence Transformers."""

from typing import Dict
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


def extract_movie_embeddings(
    token_to_title: Dict[int, str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = "cpu"
) -> np.ndarray:
    """Extracts dense sentence embeddings for movie titles.

    Args:
        token_to_title: Dict mapping mapped token ID (int) to movie title (str).
          Index 0 is assumed to be the padding token.
        model_name: Name of the Hugging Face model to use.
        batch_size: Batch size for embedding extraction.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        embeddings: A numpy array of shape (num_items + 1, embedding_dim).
          Index 0 is all zeros (padding embedding).
    """
    print(f"Loading tokenizer and model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    num_items = max(token_to_title.keys())
    # Determine embedding dimension
    dummy_input = tokenizer("dummy", return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_output = model(**dummy_input)
    embedding_dim = dummy_output.last_hidden_state.shape[-1]
    
    print(f"Extracting embeddings for {num_items} items (dim={embedding_dim})...")
    embeddings = np.zeros((num_items + 1, embedding_dim), dtype=np.float32)

    # Prepare list of titles in contiguous order (excluding padding 0)
    titles = [token_to_title.get(i, "") for i in range(1, num_items + 1)]

    # Batch extraction
    for start_idx in tqdm(range(0, len(titles), batch_size), desc="Encoding titles"):
        batch_titles = titles[start_idx : start_idx + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_titles,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        # Encode
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        # Place into target array
        end_idx = start_idx + len(batch_titles)
        # Store in i+1 slots
        embeddings[start_idx + 1 : end_idx + 1] = batch_embeddings

    return embeddings
