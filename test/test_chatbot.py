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

def test_decode_reply():
    bot = Chatbot()

    encoded = bot.encode_prompt("Hello!")
    input_ids = encoded["input_ids"][0].tolist()

    decoded = bot.decode_reply(input_ids)

    assert isinstance(decoded, str)
    assert "hello" in decoded.lower()
    
def test_generate_reply():
    bot = Chatbot()
    reply = bot.generate_reply("Hello there, how are you?")

    assert isinstance(reply, str)
    assert len(reply) > 0
    
def test_conversation_history():
    bot = Chatbot()

    first = bot.generate_reply("Hello")
    assert isinstance(first, str)
    assert bot.chat_history_ids is not None  # History should exist

    second = bot.generate_reply("How are you?")
    assert isinstance(second, str)
    assert bot.chat_history_ids.shape[-1] > 0  # History should grow

def test_reset_history():
    bot = Chatbot()
    bot.generate_reply("Test")
    bot.reset_history()
    assert bot.chat_history_ids is None
    
    