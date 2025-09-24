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

cotPrompt = """
Explain the offside rule in soccer.
Think step by step:
1. Describe when a player is considered offside.
2. Explain the role of the ball and the second-last defender.
3. Summarize the rule in one short sentence.
"""

response = llm.invoke(cotPrompt)
print(response.content)