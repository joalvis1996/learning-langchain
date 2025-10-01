import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)

context = """
A player is in an offside position if they are nearer to the opponent’s goal line
than both the ball and the second-last opponent at the moment the ball is played to them.
However, being in an offside position is not an offense by itself; the player must also be involved in active play.
"""



ragPrompt = f"""
[Context]
{context}

[Question]
What is the offside rule in soccer?
"""

response = llm.invoke(ragPrompt)
print(response.content)
