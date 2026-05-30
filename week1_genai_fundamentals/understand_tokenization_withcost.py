import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

# ── Block 2: Token Cost Intuition ──────────────────────────────────────────────
# Compare token efficiency: English locations vs Indian locations
# Real-world relevance: if you're building a pipeline with Indian data,
# you pay MORE tokens (and therefore more $) than equivalent English text.

location_pairs = [
    ("New York",          "Thiruvananthapuram"),
    ("Los Angeles",       "Visakhapatnam"),
    ("San Francisco",     "Chhatrapati Sambhajinagar"),
    ("Chicago",           "Bengaluru"),          # short Indian name — expect ~same
]

print("=" * 60)
print("LOCATION TOKEN COST: US vs India")
print("=" * 60)
for us, india in location_pairs:
    us_tokens    = enc.encode(us)
    india_tokens = enc.encode(india)
    ratio = len(india_tokens) / len(us_tokens)
    print(f"  {us:<22} → {len(us_tokens)} tokens")
    print(f"  {india:<22} → {len(india_tokens)} tokens  (×{ratio:.1f})")
    print()

# ── Numeric reasoning trap ─────────────────────────────────────────────────────
# Show how numbers are split — this is WHY LLMs fail at precise arithmetic

print("=" * 60)
print("NUMBER TOKENIZATION")
print("=" * 60)
numbers = ["42", "1000", "3.14159", "1234567890", "2024-01-15"]
for n in numbers:
    tokens = enc.encode(n)
    print(f"  {n:<15} → {len(tokens)} tokens: {[enc.decode([t]) for t in tokens]}")

print()

# ── Your Michelin context ──────────────────────────────────────────────────────
# Same sentence structure, different proper nouns — watch the token count shift

print("=" * 60)
print("MICHELIN CONTEXT: Token cost of your actual work data")
print("=" * 60)
sentences = [
    "Tyre plant located in New York",
    "Tyre plant located in Thiruvananthapuram",
    "Supplier audit scheduled for Chicago facility",
    "Supplier audit scheduled for Visakhapatnam facility",
]
for s in sentences:
    tokens = enc.encode(s)
    print(f"  ({len(tokens):>2} tokens) {s}")