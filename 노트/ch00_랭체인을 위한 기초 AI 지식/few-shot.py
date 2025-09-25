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

fewShotPrompt = """
You are a soccer coach. Explain rules simply.

Q: What is the handball rule in soccer?
A: It is a foul if a player deliberately touches the ball with their hand or arm.

Q: What is the penalty kick rule in soccer?
A: A penalty is awarded when a foul is committed inside the defending team’s penalty area.

Q: What is the offside rule in soccer?
A:
"""

response = llm.invoke(fewShotPrompt)
print(response.content)
