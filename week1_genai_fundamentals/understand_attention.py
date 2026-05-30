"""
================================================================================
FILE: understand_attention.py
WEEK:  Week 1 — How LLMs Actually Work
BLOCK: Block 3 — Transformers & Self-Attention
================================================================================

PURPOSE OF THIS FILE
--------------------
This file demonstrates the CORE mechanism inside every modern LLM: Self-Attention.

Before Transformers (pre-2017), models read text sequentially — one word at a time,
like a Kafka stream with a tiny buffer. By the time you reached word 50, context
from word 1 was mostly lost.

The Transformer (introduced in "Attention Is All You Need", Vaswani et al., 2017)
solved this by letting EVERY token directly look at EVERY other token simultaneously.
This is called Self-Attention.

WHY THIS MATTERS FOR YOU (Data Engineer context)
-------------------------------------------------
Think of Self-Attention as a fuzzy SQL JOIN:
  - Every token runs a SELECT against all other tokens
  - The JOIN key is learned, not fixed — the model decides what's "relevant"
  - The result is a weighted combination of information from all tokens

Example: In "The tyre plant failed inspection"
  - 'plant' should pay attention to 'tyre' (what kind of plant?) 
  - 'failed' should pay attention to 'inspection' (what failed?)
  - This is resolved by Self-Attention, not word order

THE Q / K / V COMPONENTS (mapped to DB concepts)
-------------------------------------------------
  Q (Query)  → What this token is asking:   "I'm 'plant' — what context am I in?"
  K (Key)    → What each token advertises:  "I'm 'tyre', I'm a physical object"
  V (Value)  → The actual information:      The content each token contributes

  Step 1: Compute attention score  = Q · K  (dot product = similarity measure)
  Step 2: Normalize scores         = softmax(scores)  → probabilities summing to 1
  Step 3: Weighted output          = sum(score × V)   for all tokens

  In a real GPT-4 Transformer:
    - Vectors are 1536-dimensional (not 3D as used here for clarity)
    - There are 96 attention "heads" running in parallel (Multi-Head Attention)
    - Q, K, V matrices are learned during training on trillions of tokens

WHAT THIS FILE DOES (simplified simulation)
--------------------------------------------
  - We use hand-crafted 3D vectors instead of learned 1536D vectors
  - We compute raw attention scores using dot products
  - We normalize them with softmax so they sum to 1 (interpretable as weights)
  - We print which tokens each token "attends to" most
  - This shows the CONCEPT without requiring a GPU or neural network

WHAT TO OBSERVE WHEN YOU RUN THIS
-----------------------------------
  1. 'tyre'  and 'plant'      → should attend to each other (similar Q/K vectors)
  2. 'failed' and 'inspection' → should attend to each other (similar Q/K vectors)
  3. 'The'                    → attends weakly everywhere (it's a stop word, not informative)

This is the same logic a real LLM uses — just at 512x the dimensionality.
================================================================================
"""

import math


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def dot_product(a, b):
    """
    Compute the dot product of two vectors.

    In attention, dot product measures SIMILARITY between a Query and a Key.
    A higher dot product = the Query and Key are more "aligned" = more attention.

    Example (2D):
        a = [1, 0]  (pointing right)
        b = [1, 0]  (also pointing right) → dot product = 1  (highly similar)
        b = [0, 1]  (pointing up)         → dot product = 0  (unrelated)
        b = [-1, 0] (pointing left)       → dot product = -1 (opposite)

    In a real Transformer, these vectors are 768–1536 dimensional.
    We use 3D here purely for readability.
    """
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    """
    Convert raw attention scores into probabilities that sum to 1.

    Why softmax?
      - Raw dot products can be any number (positive or negative)
      - We need weights that sum to 1 so we can do a weighted average
      - Softmax also amplifies the highest score and suppresses low scores
        (the model becomes more "decisive" about what to attend to)

    Example:
        raw scores  = [0.5, 1.2, 0.1]
        after softmax → [0.22, 0.56, 0.18]  ← sums to ~1.0

    In a real Transformer, scores are also divided by sqrt(d_k) before softmax
    to prevent very large values from making gradients vanish during training.
    (d_k = dimension of the key vectors, e.g. 64 in GPT-2)
    """
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


# ── TOKEN SETUP ───────────────────────────────────────────────────────────────

# Our sentence: "The tyre plant failed inspection"
# In a real LLM, the tokenizer first converts this to token IDs,
# then an embedding layer converts each ID to a high-dimensional vector.
# Here we skip that and directly define the tokens as strings.
tokens = ["The", "tyre", "plant", "failed", "inspection"]


# ── QUERY AND KEY VECTORS ─────────────────────────────────────────────────────
#
# In a real Transformer:
#   - Each token starts as an embedding vector (e.g. 1536-dimensional)
#   - Three learned weight matrices (Wq, Wk, Wv) transform it into Q, K, V
#   - These matrices are learned during pre-training on trillions of tokens
#
# Here we hand-craft 3D vectors to simulate what those learned matrices produce.
# The design intent:
#   - 'tyre' and 'plant' have similar Q/K vectors → they'll attend to each other
#   - 'failed' and 'inspection' have similar Q/K vectors → they'll attend to each other
#   - 'The' has a near-zero vector → weak attention everywhere (stop word)

queries = {
    # token       [dim1, dim2, dim3]
    "The":        [0.1,  0.2,  0.0],   # weak query — stop word, low semantic weight
    "tyre":       [0.9,  0.1,  0.2],   # strong on dim1 — physical object signal
    "plant":      [0.8,  0.3,  0.1],   # similar to 'tyre' on dim1 → will attend to tyre
    "failed":     [0.2,  0.9,  0.4],   # strong on dim2 — event/action signal
    "inspection": [0.1,  0.8,  0.5],   # similar to 'failed' on dim2 → will attend to failed
}

keys = {
    # token       [dim1, dim2, dim3]
    "The":        [0.0,  0.1,  0.1],   # advertises very little
    "tyre":       [0.8,  0.2,  0.1],   # advertises: "I'm a physical object"
    "plant":      [0.7,  0.2,  0.2],   # advertises: "I'm also a physical object"
    "failed":     [0.3,  0.8,  0.3],   # advertises: "I'm an event/action"
    "inspection": [0.2,  0.7,  0.6],   # advertises: "I'm related to an event"
}

# Note: In production, Q and K are typically different projections of the same
# input vector. We keep them separate here to make the "asking vs advertising"
# intuition explicit.


# ── COMPUTE AND DISPLAY ATTENTION ─────────────────────────────────────────────

print("=" * 60)
print("SELF-ATTENTION: Which tokens does each token attend to?")
print("=" * 60)
print("(Higher score = more attention paid to that token)\n")

for token in tokens:
    q = queries[token]

    # Step 1: Compute raw attention score of this token against ALL other tokens
    # This is Q · K for every (token, other_token) pair
    raw_scores = [dot_product(q, keys[other]) for other in tokens]

    # Step 2: Normalize with softmax → attention weights sum to 1
    attention_weights = softmax(raw_scores)

    # Step 3: Display as a bar chart for intuitive reading
    # (In a real Transformer, Step 3 would be: output = sum(weight × Value vector))
    print(f"Token: '{token}' attends to:")
    for other, weight in zip(tokens, attention_weights):
        bar = "█" * int(weight * 30)   # bar length proportional to attention weight
        print(f"  {other:<12} {weight:.3f}  {bar}")
    print()

print("=" * 60)
print("KEY INSIGHT:")
print("'plant' and 'tyre' have similar Q/K vectors → attend to each other more")
print("'failed' and 'inspection' have similar Q/K vectors → attend to each other more")
print("This is how context is built — not from word order alone, but from learned relevance.")
print()
print("In a real GPT-4 call, this computation runs for EVERY token pair,")
print("across 96 attention heads, for each of the 96 Transformer layers.")
print("That's what makes LLMs expensive to run — and powerful.")
