import os
from enum import Enum

from dotenv import load_dotenv
from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# 환경 변수 로드
load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)


# Enum 정의
class Level(Enum):
    WORLDCLASS = "세계 최고의 선수"
    INTERNATONAL = "국제적인 선수"
    NATIONAL = "국내 최고의 선수"


parser = EnumOutputParser(enum=Level)

prompt = PromptTemplate(
    template=(
        "다음 축구 선수의 실력을 아래 지침대로 분류하세요.\n"
        "{instructions}\n\n"
        "선수: {player}\n"
        "정답:"
    ),
    input_variables=["player"],
    partial_variables={"instructions": parser.get_format_instructions()}
)
print(parser.get_format_instructions())
exit

chain = prompt | llm | parser

response = chain.invoke({"player": "빅터 요케레스"})

print(response)