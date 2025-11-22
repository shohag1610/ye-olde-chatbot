from transformers import AutoTokenizer, AutoModelForCausalLM


class Chatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        # Add newline for better generation behaviour
        prompt = prompt + "\n"

        encoded = self.encode_prompt(prompt)

        output_ids = self.model.generate(
            **encoded,
            max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=50
        )

        # Convert tensor to Python list
        generated_ids = output_ids[0].tolist()

        # Decode the entire sequence
        decoded = self.decode_reply(generated_ids)

        # Remove original prompt to get only new reply
        reply = decoded.replace(prompt.strip(), "", 1).strip()

        return reply