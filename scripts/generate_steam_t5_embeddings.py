import os
import json
import gzip
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_steam_text():
    print("Loading item map from steam cache...")
    with open("./data/steam/steam_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    item_map = cache['item_to_id']
    
    num_items = len(item_map)
    # The padding item (ID=0) will be all zeros.
    # The items start from ID=1 to num_items. Total size = num_items + 1.
    print(f"Total valid items in cache: {num_items}. Generating embeddings for size {num_items + 1}")
    
    # Store text
    texts = [""] * (num_items + 1)
    
    print("Reading steam_games.json.gz...")
    found_count = 0
    with gzip.open("./data/steam/steam_games.json.gz", "rt") as f:
        for line in f:
            try:
                game = eval(line.strip()) # data is in eval-able python dict format
            except:
                continue
            
            app_id = game.get('id', None)
            if not app_id:
                continue
                
            if app_id in item_map:
                idx = item_map[app_id]
                title = game.get('title', game.get('app_name', ''))
                genres = ", ".join(game.get('genres', []))
                tags = ", ".join(game.get('tags', []))
                
                text_repr = f"{title}. Genres: {genres}. Tags: {tags}"
                texts[idx] = text_repr
                found_count += 1
                
    print(f"Found metadata for {found_count} out of {num_items} items.")
    
    # Fill missing with empty string to avoid errors
    
    print("Loading sentence-transformers/sentence-t5-base...")
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    
    print("Computing embeddings... This might take a minute.")
    # Calculate embeddings in batches
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    out_path = "./data/steam_t5_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    extract_steam_text()
