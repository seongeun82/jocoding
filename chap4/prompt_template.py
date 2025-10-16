from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt_template = "이 음식 리뷰 '{review}' 에대해서 '{rating1}' 점부터 '{rating2}' 점까지의 점수로 평가해줘."

prompt = PromptTemplate(
    input_variables=["review", "rating1", "rating2"], template=prompt_template
)

openai = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

chain = prompt | openai | StrOutputParser()


def main():
    print("Chain created successfully.")
    try:
        response = chain.invoke(
            {
                "review": "이 음식은 맛있었어요! 우리집 개가 좋아할 것 같아요 재료가 신선하고 조리만 훌륭했어요.",
                "rating1": 1,
                "rating2": 5,
            }
        )
        print(f"Response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
