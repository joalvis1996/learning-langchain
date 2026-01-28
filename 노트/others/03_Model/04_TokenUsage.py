import json

from dotenv import load_dotenv
from langchain_core.load import load

load_dotenv()

with open("geminiLLM.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    


llm = load(data)

response = llm.invoke("너는 누구니")
print(response.usage_metadata)

## 실행 결과
# {'input_tokens': 5, 'output_tokens': 13, 'total_tokens': 115, 'input_token_details': {'cache_read': 0}}
