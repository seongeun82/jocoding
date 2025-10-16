import os
from dotenv import load_dotenv
from langchain.chains.sequential import SequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# 1. 리뷰 한글로 번역 
#2. 리뷰요약
#3. 점수화 
#4. 언어감지 
#5. 답변생성 
#6. 답변 번역

prompt1 = PromptTemplate(
    input_variables=["review"],
    template="다음 리뷰를 한글로 번역해줘: {review}"
)

