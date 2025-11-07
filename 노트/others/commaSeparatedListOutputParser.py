import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser


# 환경 변수 로드
load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template="{topic}에 관련된 용어 5가지. "
             "\n{format_instructions}",
    input_varialbes=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

response = chain.invoke({"topic": "영국 프리미어 리그"})
print(response)

## 실행 결과
## ['프리미어 리그', '강등', '탑 4', '이적 시장', '더비']
 