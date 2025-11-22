from chatbot import Chatbot

def main():
    bot = Chatbot()
    prompt = "What is the weather like today?"
    reply = bot.generate_reply(prompt)

    print(f"Prompt: {prompt}")
    print(f"Reply: {reply}")

if __name__ == "__main__":
    main()
