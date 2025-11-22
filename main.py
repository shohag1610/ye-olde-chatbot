from chatbot import Chatbot

def main():
    bot = Chatbot()
    encoded = bot.encode_prompt("Hello, how are you?")
    print(encoded)

if __name__ == "__main__":
    main()
