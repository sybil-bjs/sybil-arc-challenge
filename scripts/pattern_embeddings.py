#!/usr/bin/env python3
"""
Semantic search for patterns using embeddings.

Uses TF-IDF for lightweight local embeddings (no API key needed).
Can upgrade to Gemini/OpenAI embeddings if API keys available.

Usage:
  python pattern_embeddings.py index          # Build embeddings for all patterns
  python pattern_embeddings.py search <query> # Semantic search
  python pattern_embeddings.py similar <task_id> # Find patterns for a task
"""

import os
import sys
import json
import sqlite3
import numpy as np
from pathlib import Path
from collections import Counter
import re
import math

REPO = Path.home() / "sybil-arc-challenge"
DB_PATH = REPO / "knowledge" / "patterns.db"
EMBEDDINGS_PATH = REPO / "knowledge" / "embeddings.npz"
VOCAB_PATH = REPO / "knowledge" / "vocab.json"

def tokenize(text: str) -> list:
    """Simple tokenization."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(texts: list) -> dict:
    """Build vocabulary from texts."""
    vocab = {}
    doc_freq = Counter()
    
    for text in texts:
        tokens = set(tokenize(text))
        for token in tokens:
            doc_freq[token] += 1
    
    # Filter rare/common tokens and build vocab
    n_docs = len(texts)
    for i, (token, freq) in enumerate(sorted(doc_freq.items())):
        if freq >= 1 and freq < n_docs:  # Not too rare, not in every doc
            vocab[token] = len(vocab)
    
    return vocab

def tfidf_embedding(text: str, vocab: dict, idf: dict) -> np.ndarray:
    """Compute TF-IDF embedding for text."""
    tokens = tokenize(text)
    tf = Counter(tokens)
    
    vec = np.zeros(len(vocab))
    for token, count in tf.items():
        if token in vocab:
            idx = vocab[token]
            vec[idx] = (1 + math.log(count)) * idf.get(token, 1.0)
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec

def get_embedding(text: str, vocab: dict = None, idf: dict = None) -> np.ndarray:
    """Get embedding for text using TF-IDF."""
    if vocab is None:
        # Load vocab
        if VOCAB_PATH.exists():
            with open(VOCAB_PATH) as f:
                data = json.load(f)
                vocab = data['vocab']
                idf = data['idf']
        else:
            raise ValueError("Vocab not built. Run: python pattern_embeddings.py index")
    
    return tfidf_embedding(text, vocab, idf)

def get_query_embedding(text: str) -> np.ndarray:
    """Get embedding for a search query."""
    return get_embedding(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_index():
    """Build TF-IDF embeddings for all patterns in the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    patterns = conn.execute("""
        SELECT id, name, description, keywords FROM patterns
    """).fetchall()
    
    if not patterns:
        print("No patterns to index.")
        return
    
    print(f"Building TF-IDF embeddings for {len(patterns)} patterns...")
    
    # Collect all texts
    texts = []
    for p in patterns:
        text = f"{p['name']} {p['description']} {p['keywords']}"
        texts.append(text)
    
    # Build vocabulary
    vocab = build_vocab(texts)
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Compute IDF
    n_docs = len(texts)
    doc_freq = Counter()
    for text in texts:
        tokens = set(tokenize(text))
        for token in tokens:
            if token in vocab:
                doc_freq[token] += 1
    
    idf = {token: math.log(n_docs / (1 + freq)) for token, freq in doc_freq.items()}
    
    # Save vocab
    with open(VOCAB_PATH, 'w') as f:
        json.dump({'vocab': vocab, 'idf': idf}, f)
    
    # Build embeddings
    embeddings = {}
    for p, text in zip(patterns, texts):
        emb = tfidf_embedding(text, vocab, idf)
        embeddings[p['id']] = {
            'name': p['name'],
            'embedding': emb
        }
        print(f"  ✓ {p['name']}")
    
    # Save embeddings
    np.savez(
        EMBEDDINGS_PATH,
        ids=np.array([e for e in embeddings.keys()]),
        names=np.array([embeddings[e]['name'] for e in embeddings]),
        vectors=np.array([embeddings[e]['embedding'] for e in embeddings])
    )
    
    print(f"✅ Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")

def load_index():
    """Load embeddings from disk."""
    if not EMBEDDINGS_PATH.exists():
        return None
    
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return {
        'ids': data['ids'],
        'names': data['names'],
        'vectors': data['vectors']
    }

def semantic_search(query: str, top_k: int = 5) -> list:
    """Search patterns by semantic similarity."""
    index = load_index()
    if index is None:
        print("No embeddings index. Run: python pattern_embeddings.py index")
        return []
    
    # Get query embedding
    query_emb = get_query_embedding(query)
    
    # Compute similarities
    similarities = []
    for i, vec in enumerate(index['vectors']):
        sim = cosine_similarity(query_emb, vec)
        similarities.append((index['ids'][i], index['names'][i], sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities[:top_k]

def describe_task(task_id: str) -> str:
    """Generate a text description of a task for embedding."""
    sys.path.insert(0, str(REPO / "scripts"))
    from suggest_pattern import load_task, extract_features
    
    task = load_task(task_id)
    features = extract_features(task)
    
    # Build description
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    
    desc = f"ARC puzzle with input size {len(inp)}x{len(inp[0])} and output size {len(out)}x{len(out[0])}. "
    desc += f"Features: {', '.join(f'{k}={v}' for k,v in features.items())}. "
    
    # Add color info
    import numpy as np
    colors = set(np.unique(inp)) - {0}
    desc += f"Input colors: {colors}. "
    
    return desc

def find_similar_for_task(task_id: str, top_k: int = 5) -> list:
    """Find patterns similar to a given task."""
    desc = describe_task(task_id)
    print(f"Task description: {desc[:100]}...")
    return semantic_search(desc, top_k)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pattern_embeddings.py <index|search|similar> [args]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "index":
        build_index()
    
    elif cmd == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"\nSearching for: {query}\n")
        results = semantic_search(query)
        if results:
            print("Results:")
            for id_, name, score in results:
                print(f"  {score:.3f}  {name}")
        else:
            print("No results.")
    
    elif cmd == "similar" and len(sys.argv) > 2:
        task_id = sys.argv[2]
        print(f"\nFinding patterns similar to task {task_id}\n")
        results = find_similar_for_task(task_id)
        if results:
            print("Similar patterns:")
            for id_, name, score in results:
                print(f"  {score:.3f}  {name}")
        else:
            print("No similar patterns found.")
    
    else:
        print("Usage: python pattern_embeddings.py <index|search|similar> [args]")
