from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI()
prompt = [HumanMessage('프랑스의 수도는 어디인가요?')]

response = model.invoke(prompt)
print(response.content)
