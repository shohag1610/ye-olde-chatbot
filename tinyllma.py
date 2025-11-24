import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Chatbot:
    def __init__(self, model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        self.system_prompt = "<|system|>\nYou are a helpful AI assistant.<|end|>\n"

    def clean_prompt_for_printing(self, text):
        return (
            text.replace("<|system|>", "")
            .replace("<|user|>", "")
            .replace("<|assistant|>", "")
            .replace("<|end|>", "")
            .strip()
        )

    def encode_prompt(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def decode_reply(self, output_ids):
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    def generate_single_reply(self, prompt: str):
        prompt = prompt.strip()
        inputs = self.encode_prompt(prompt)

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.9,
        )

        reply = self.decode_reply(output_ids[0][inputs["input_ids"].size(1) :]).strip()
        return reply

    def generate_reply(self, prompt: str) -> str:
        prompt = prompt.strip()

        if not hasattr(self, "conversation_history"):
            self.conversation_history = (
                self.system_prompt
                # + "\n<|user|>\nHello, how art thou today?<|end|>\n<|assistant|>\nVerily, I am well, kind soul. How fare thee?<|end|>\n<|user|>\nWhat thinkest thou of the weather?<|end|>\n<|assistant|>\nThe heavens weep or smile, as doth the mood of fate. 'Tis fair today, by mine eye.<|end|>\n"
            )

        self.conversation_history += f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        inputs = self.encode_prompt(self.conversation_history)

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.9,
        )

        new_tokens = output_ids[0][inputs["input_ids"].size(1) :]
        reply = self.decode_reply(new_tokens).strip()

        self.conversation_history += f"{reply}<|end|>\n"

        return reply

    def reset_history(self):
        self.conversation_history = self.system_prompt