# 🤖 Ye Olde Chatbot

A conversational **AI chatbot** built using **Python**, **PyTorch**, and **HuggingFace Transformers**.  
Supports GPU acceleration, conversation history, system prompts, and context-aware responses.

---

## 🚀 Features

- 💬 Context-aware responses with conversation history  
- ⚡ GPU-accelerated inference using PyTorch  
- 🧠 System prompt support to guide bot behavior  
- 🛠 Easy to extend to other LLMs (DialoGPT, GPT-Neo, Falcon, etc.)  
- 🖥️ Terminal-based interactive chat UI  

---

## 📦 Tech Stack

- Python 3.10+  
- PyTorch  
- HuggingFace Transformers  
- Optional GPU support  

---

## 🔧 Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/ye-olde-chatbot.git
cd ye-olde-chatbot
```

# Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

# Install dependencies
```bash
pip install torch transformers rich pytest
```
---

## ▶️ How to Run
After installing dependencies, run the chatbot in the terminal:

```bash
python main.py
```

## ▶️ How to use
The chatbot will start and display a system prompt. You can then type messages interactively:

User: Hello!  
Bot: Greetings! How can I assist you today?  

Type your message after **User:**  
Bot replies are printed as **Bot:**  
Type **quit** or **exit** to stop the conversation.

---

## 🧠 How to Change Model
You can use different HuggingFace models by specifying the model_name when creating the Chatbot object inside main.py:
```bash
from chatbot import Chatbot
bot = Chatbot(model_name="microsoft/DialoGPT-medium")
```
# Example: Using a different model
- Supported Models
- microsoft/DialoGPT-small
- microsoft/DialoGPT-medium
- microsoft/DialoGPT-large
- facebook/opt-350m
- tiiuae/falcon-rw-1b
- facebook/opt-1.3b
- EleutherAI/gpt-neo-1.3B

Note: Larger models may require GPU for faster inference.
Adjust generation parameters like max_new_tokens, temperature, top_p, and top_k as needed.