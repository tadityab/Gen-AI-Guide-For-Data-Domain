import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

texts = [
    "Hello world",
    "tokenization",
    "Thiruvananthapuram",
    "def get_user(id):",
    "1234567890",
    "I am a Senior Data Engineer at Google India",
    "sham is Senior Data Engineer at Google India"
]

for text in texts:
    tokens = enc.encode(text)
    print(f"Text: {text!r}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Tokens: {[enc.decode([t]) for t in tokens]}")
    print("---")