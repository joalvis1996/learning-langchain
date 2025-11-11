import os

from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=apiKey
)


class MovieInfo(BaseModel):
    title: str = Field(description="영화 제목")
    director: str = Field(description="감독 이름")
    releaseYear: int = Field(description="개봉 연도")


parser = PydanticOutputParser(pydantic_object=MovieInfo)

# 잘못된 형식을 일부러 입력
misFormattedResult = "{'title': 'Tom Hanks', 'director': 'Forrest Gump', 'releaseYear': 2025}"

# 에러 발생: Invalid json output: {'title': 'Tom Hanks', 'director': 'Forrest Gump', 'releaseYear': 2025}
# parser.parse(misFormattedResult)

fixingParser = OutputFixingParser.from_llm(parser=parser, llm=llm)

movie = fixingParser.parse(misFormattedResult)
print(movie)

## 실행 결과
# title='Forrest Gump' director='Robert Zemeckis' releaseYear=1994