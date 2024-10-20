# pip install langchain
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi

import os

os.environ["DASHSCOPE_API_KEY"] = "sk-8578ff9997934be48b142b5996a47927"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm_tongyi = Tongyi()

vector_template = """
背景是：我要用一个三个元素的向量，来表示自动驾驶过程中的安全性、舒适性、效率性权重特征
问题是: {question}，请根据我的需求，为我生成一个三个元素的向量，三个元素分别表示有以下几点要求
- 三个元素表示：安全性、舒适性、效率性
- 三个元素的和是1
- 三个元素都保留小数点后2位
- 安全性不低于0.70
具体取值请根据我的需求进行判断，只需要输出这一个向量，不要有其他多余的内容"""

analysis_template = """
背景是：我要用一个三个元素的向量，来表示自动驾驶过程中的安全性、舒适性、效率性权重特征
这个向量是：{vector}
问题是: {question}，请根据我的需求。分析一下为什么向量这么取值"""

vector_prompt = PromptTemplate.from_template(vector_template)
analysis_prompt = PromptTemplate.from_template(analysis_template)

vector_chain = vector_prompt | llm_tongyi
analysis_chain = analysis_prompt | llm_tongyi

def query_vector(message: str):
    vector_response = vector_chain.invoke({"question": message})
    return vector_response

def analyze_vector(vector: str, message: str):
    analysis_response = analysis_chain.invoke({"vector": vector, "question": message})
    return analysis_response

def get_response(message: str):
    vector = query_vector(message)
    explanation = analyze_vector(vector, message)
    return vector, explanation

def query_vector(message: str):
    vector_response = vector_chain.invoke({"question": message})
    # 解析向量字符串，比如 "[0.70, 0.10, 0.20]" -> [0.70, 0.10, 0.20]
    vector = vector_response.strip('[]').split(',')
    vector = [float(val) for val in vector]  # 转换为浮点数数组
    return vector

if __name__ == "__main__":
    message = "我赶时间，想要开的快一些，减少舒适性的考虑"
    vector, explanation = get_response(message)
    print(vector)
    print(explanation)