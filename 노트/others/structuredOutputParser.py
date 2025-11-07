import os

from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
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

schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(name="source", description="사용자 질문에 답하기 위해 사용된 출처(웹사이트주소)"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

prompt = PromptTemplate(
    template="사용자의 질문에 답변해줘.\n"
             "{format_instructions}\n"
             "내용:{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
response = chain.invoke({"question": "EPL의 아스날 역대 득점 1위 선수는 누구인가요"})

print(response)

# 실행 결과
# {'answer': 'EPL의 아스날 역대 득점 1위 선수는 티에리 앙리(Thierry Henry)입니다. 그는 아스날에서 총 228골을 기록했습니다.', 'source': 'https://www.arsenal.com/club/history/greatest-players/thierry-henry'}
 