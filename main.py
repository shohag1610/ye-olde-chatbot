from chatbot import Chatbot

def main():
    bot = Chatbot()
    encoded = bot.encode_prompt("Hello, how are you?")
    ids = encoded["input_ids"][0].tolist()

    print("Token IDs:", ids)
    print("Decoded:", bot.decode_reply(ids))

if __name__ == "__main__":
    main()
