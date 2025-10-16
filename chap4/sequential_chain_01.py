from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# sequential_chain으로 해결하기
# 1. 리뷰요약
# 2. 리뷰점수화
# 3. 리뷰 응답 메시지 작성

prompt1 = PromptTemplate.from_template("다음 가게 리뷰를 한문장으로 요약해줘: {review}")
chain1 = LLMChain(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.7),
    prompt=prompt1,
    output_key="summary",
)
prompt2 = PromptTemplate.from_template(
    "다음 리뷰를 1점부터 5점까지 점수로 평가해줘: {summary}"
)
chain2 = LLMChain(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.7),
    prompt=prompt2,
    output_key="score",
)

prompt3 = PromptTemplate.from_template(
    "다음 리뷰를 바탕으로 응답 메시지를 작성해줘 요약 메시지와 점수에 대해서도 언급해줘: {review}, 요약: {summary}, 점수: {score}"
)
chain3 = LLMChain(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.7),
    prompt=prompt3,
    output_key="response",
)

sequential_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["review"],
    output_variables=["summary", "score", "response"],
)

review_text = "이 식당은 맛도 좋고 분위기도 좋았습니다. 가격 대비 만족도가 높아요. 하지만, 서비스 속도가 너무 느려서 조금 실망스러웠습니다. 전반적으로 다시 방문할 의향이 있습니다. 사실 거짓말 다시오지 않을거임. 최악"

try:
    result = sequential_chain.invoke({"review": review_text})
    print("Summary:", result["summary"])
    print("Score:", result["score"])
    print("Response:", result["response"])
except Exception as e:
    print(f"An error occurred: {e}")
