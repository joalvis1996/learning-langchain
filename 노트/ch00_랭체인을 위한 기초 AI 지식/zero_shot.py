import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
apiKey = os.getenv("google_api_key")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)

zeroShotPrompt = "Explain the offside rule in soccer"
response = llm.invoke(zeroShotPrompt)

print(response.content)
