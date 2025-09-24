# 기존에 정의해놓은 프롬프트 템플릿 가져오기
from langchain.prompts import PromptTemplate              
from dotenv import load_dotenv

# Google Gemini(Generative AI)용 LangChain 래퍼(모델 어댑터). LangChain 표준 인터페이스로 Gemini를 호출할 수 있게 해줌.
from langchain_google_genai import ChatGoogleGenerativeAI 

import os

load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=apiKey
)

template = """
당신은 선생님입니다. 
학생이 '{topic}'에 대해 묻습니다. 친절하게 설명해주세요.
"""

prompt = PromptTemplate.from_template(template)   # 문자열을 LangChain이 이해할 수 있는 PromptTemplate 객체로 변환하는 메서드
finalPrompt = prompt.format(topic="축구에서의 오프사이드 규칙")

response = llm.invoke(finalPrompt)
print(response.content)