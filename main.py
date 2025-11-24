from chatbot import Chatbot

def main():
    bot = Chatbot()

    print("===============================================")
    print("      Welcome to Your Terminal Chatbot!        ")
    print("===============================================")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        bot_reply = bot.generate_reply(user_input)
        print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    main()
