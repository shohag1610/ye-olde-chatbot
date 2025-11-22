from transformers import AutoTokenizer, AutoModelForCausalLM


class Chatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
