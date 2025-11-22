from chatbot import Chatbot

def test_chatbot_initialises():
    bot = Chatbot()

    assert bot.model_name == "microsoft/DialoGPT-small"
    assert bot.tokenizer is not None
    assert bot.model is not None
    
def test_encode_prompt_returns_tensors():
    bot = Chatbot()
    encoded = bot.encode_prompt("Hello!")

    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    assert encoded["input_ids"].shape[0] == 1
    assert encoded["attention_mask"].shape[0] == 1
