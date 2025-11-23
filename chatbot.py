from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Chatbot:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chatbot running on: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        #to store conversation history
        self.chat_history_ids = None
        
        #default
        self.system_prompt = "You are a helpful assistant. Respond to the end of this conversation accordingly.\n"
        
    def encode_prompt(self, prompt: str):
        prompt = prompt + "\n"
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def reset_history(self):
        self.chat_history_ids = None
    
    def generate_reply(self, prompt: str) -> str:

        if self.chat_history_ids is None:
            full_prompt = self.system_prompt + prompt
        else:
            full_prompt = prompt
            
        # Encode the prompt
        encoded = self.encode_prompt(full_prompt)

        # If no history → first message
        if self.chat_history_ids is None:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            # Concatenate previous history with new prompt
            input_ids = torch.cat([self.chat_history_ids, encoded["input_ids"]], dim=-1).to(self.device)

            # Build attention mask to match the concatenated input_ids
            attention_mask = torch.cat(
                [
                    torch.ones_like(self.chat_history_ids),
                    encoded["attention_mask"]
                ],
                dim=-1
            ).to(self.device)

        # Generate model output
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 100,
            pad_token_id=self.tokenizer.eos_token_id,  
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            top_k=50
        )

        # Extract ONLY newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]

        # Decode the reply text
        reply = self.decode_reply(new_tokens.tolist()).strip()

        # Update the chat history (the full conversation output_ids)
        self.chat_history_ids = output_ids.to(self.device)

        return reply