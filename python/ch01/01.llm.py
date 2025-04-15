from langchain_openai.llms import OpenAI

model = OpenAI(model='gpt-3.5-turbo-instruct')

response = model.invoke('하늘이')
print(response)
