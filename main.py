from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="deepseek-r1:1.5b")

template = """

you are a helpful assistant that can answer questions and help with tasks.

here are some relevant reviews : {reviews}

here is the answer to the question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
result = chain.invoke({"reviews": [], "question": "what is the best product in the market?"})
print(result)





