from chatbot import Chatbot

def test_chatbot_initialises():
    bot = Chatbot()

    assert bot.model_name == "microsoft/DialoGPT-small"
    assert bot.tokenizer is not None
    assert bot.model is not None

