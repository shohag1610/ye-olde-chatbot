from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Chatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-large"):
        self.model_name = model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chatbot running on: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        #to store conversation history
        self.chat_history_ids = None
        
        #store prompt history as a text
        self.text_history = "" 
        
        #default
        self.system_prompt = "You are a helpful AI assistant.\n"
        
    def encode_prompt(self, prompt: str):
        prompt = prompt + "\n"
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def reset_history(self):
        self.chat_history_ids = None
    
    def generate_reply(self, prompt: str) -> str:
        # Add the user message to text history
        self.text_history += f"User: {prompt}\n"

        # Full prompt = system prompt only if first message + current text history
        if self.chat_history_ids is None:
            full_prompt = self.system_prompt + self.text_history
        else:
            full_prompt = self.text_history

        # Encode
        encoded = self.encode_prompt(full_prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]


        # Generate
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=50
        )

        # Extract only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        reply = self.decode_reply(new_tokens.tolist()).strip()
        reply = reply.split("User :")[-1].strip()
        self.text_history += f"Bot: {reply}\n"

        return reply