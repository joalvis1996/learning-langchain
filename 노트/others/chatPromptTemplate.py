from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="google-gemini-2.5",
    temperature=0.2,
    apiKey=apiKey
)

chatPrompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 선생님입니다. 모든 답변은 초보자도 이해가기 쉽게 설명해주세요."),
    ("user", "학생이 {topic}에 대해 질문합니다. 쉽게 설명해주세요.")
])

chatPrompt.format_messages()
