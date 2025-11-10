import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)


# 출력 스키마: Pydantic 모델 정의
class FootBallPlayerInfo(BaseModel):
    name: str = Field(description="이름")
    birthday: str = Field(description="생년월일")
    club: str = Field(description="소속 팀")
    nationality: str = Field(description="국적")


# OutputParser 생성
parser = PydanticOutputParser(pydantic_object=FootBallPlayerInfo)

# PromptTemplate 정의
prompt = PromptTemplate(
    template=(
        "다음 문장을 {format_instructions}에 맞는 JSON으로 변환해줘.\n"
        "{format_instructions}\n\n"
        "문장: {sentence}"
    ),
    input_variables=["sentence"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# Chain 연결 (Prompt → LLM → Parser)
chain = prompt | llm | parser

# 실행
sentence = "부카요 사카는 잉글랜드 프로 축구 선수로, 현재 잉글랜드 프리미어리그의 아스널 FC에서 윙어로 활약하고 있으며, 잉글랜드 축구 국가대표팀 소속이기도 합니다. 2001년 9월 5일생인 그는 " \
           "드리블, 창의성, 활동량으로 유명하며, 세계적인 선수 중 한 명으로 평가받고 있습니다."
response = chain.invoke({"sentence": sentence})
print(response)


# 실행 결과
# name='부카요 사카' birthday='2001년 9월 5일' club='아스널 FC' nationality='잉글랜드'
 