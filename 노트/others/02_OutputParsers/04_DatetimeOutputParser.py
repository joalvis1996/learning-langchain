import os

from dotenv import load_dotenv
from langchain.output_parsers import DatetimeOutputParser
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

parser = DatetimeOutputParser()

prompt = PromptTemplate(
    template="{question}\n"
    "{instructions}",
    input_variables=["question"],
    partial_variables={"instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

response = chain.invoke({"question": "미켈 아르테타가 아스날에 부임한 날짜는?"})
print(type(response))
print(response)

## 실행 결과
# <class 'datetime.datetime'>
# 2019-12-20 00:00:00