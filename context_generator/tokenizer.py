from transformers import AutoTokenizer

MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print("Tokenizer loaded successfully!")
