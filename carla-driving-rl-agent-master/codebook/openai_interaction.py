from openai import OpenAI

# 生成向量的模板
vector_template = """
背景是：我要用一个三个元素的向量，来表示自动驾驶过程中的安全性、舒适性、效率性权重特征
问题是: {question}，请根据我的需求，为我生成一个三个元素的向量，三个元素分别表示有以下几点要求
- 三个元素表示：安全性、舒适性、效率性
- 三个元素的和是1
- 三个元素都保留小数点后2位
- 安全性不低于0.70
具体取值请根据我的需求进行判断，只需要输出这一个向量，不要有其他多余的内容
"""

# 分析向量的模板
analysis_template = """
背景是：我要用一个三个元素的向量，来表示自动驾驶过程中的安全性、舒适性、效率性权重特征
这个向量是：{vector}
问题是: {question}，请根据我的需求。分析一下为什么向量这么取值
"""

client = OpenAI(
    api_key="sk-8578ff9997934be48b142b5996a47927", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
)

def generate_response(prompt: str) -> str:
    """ 使用 OpenAI API 生成模型响应 """
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content

def query_vector(message: str) -> str:
    """ 生成向量 """
    prompt = vector_template.format(question=message)
    return generate_response(prompt)

def analyze_vector(vector: str, message: str) -> str:
    """ 分析向量 """
    prompt = analysis_template.format(vector=vector, question=message)
    return generate_response(prompt)

def get_response(message: str):
    """ 获取向量及其分析 """
    vector = query_vector(message)
    explanation = analyze_vector(vector, message)
    return vector, explanation

if __name__ == "__main__":
    message = "我赶时间，想要开的快一些，减少舒适性的考虑"
    vector, explanation = get_response(message)
    print("生成的向量:", vector)
    print("向量分析:", explanation)
