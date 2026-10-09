from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Explain concepts clearly and simply."
    ),
    (
        "human",
        "{question}"
    )
])

chain = prompt | model


print("================================")
print("      LangChain AI Chatbot")
print("================================")
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = chain.invoke({
        "question": user_input
    })

    print("AI:", response.content)