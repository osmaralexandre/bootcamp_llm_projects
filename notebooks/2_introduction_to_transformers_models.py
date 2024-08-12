# %%
import openai
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

# Config API Key to the OpenAI library
openai.api_key = api_key

# conversation history
conversation_history = [
    {
        "role": "system",
        "content": "Você é um assistente que ajuda com informações sobre investimentos financeiros.",
    }
]


def ask_chatgpt(question):
    # Add questions to the history
    conversation_history.append({"role": "user", "content": question})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=150,
        )
        answer = response.choices[0].message["content"].strip()

        # Add answer to the history
        conversation_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    while True:
        user_question = input("Type your question to ChatGPT (or 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        answer = ask_chatgpt(user_question)
        print("Answer of ChatGPT:", answer)
# %%
