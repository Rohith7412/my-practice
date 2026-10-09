from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid mathematical expression."


@tool
def get_weather(city: str) -> str:
    """Get the weather of a city."""

    weather_data = {
        "hyderabad": "32°C, Sunny",
        "delhi": "30°C, Cloudy",
        "mumbai": "28°C, Rainy",
        "bangalore": "25°C, Pleasant"
    }

    return weather_data.get(city.lower(),"Weather information not available.")


tools = [calculator,get_weather]

model_with_tools = model.bind_tools(tools)

print("==============================")
print("      Tool Chatbot")
print("==============================")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = model_with_tools.invoke(user_input)

    # Check whether the model wants to call a tool
    if response.tool_calls:

        for tool_call in response.tool_calls:

            print("Tool:", tool_call["name"])
            print("Arguments:", tool_call["args"])

    else:
        print("AI:", response.content)