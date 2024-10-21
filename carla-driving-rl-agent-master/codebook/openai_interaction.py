# edit 10.20    实现 文字要求输入  输出权重vector
# improve 10.21  实现 carla自车和周车信息 + 文字 输入 输出权重vector
# 并预留 图片接口
from openai import OpenAI
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os


# 生成向量的模板
vector_template = """
背景是：我要用一个三个元素的向量，来表示自动驾驶过程中的安全性、舒适性、效率性权重特征
自车的位置是：{ego_position}
周围车辆的位置是：{surrounding_positions}
问题是: {question}，请根据我的需求，为我生成一个三个元素的向量，三个元素分别表示有以下几点要求
- 三个元素表示：安全性、舒适性、效率性
- 三个元素的和是1
- 三个元素都保留小数点后2位
- 安全性不低于0.50
具体取值请根据我的需求进行判断，只需要输出这一个向量，不要有其他多余的内容
"""

# 分析向量的模板
analysis_template = """
背景是：我要用一个三个元素的向量，来P表示自动驾驶过程中的安全性、舒适性、效率性权重特征
自车的位置是：{ego_position}
周围车辆的位置是：{surrounding_positions}
这个向量是：{vector}
问题是: {question}，请根据我的需求。分析一下为什么向量这么取值
"""

client = OpenAI(
    api_key="sk-8578ff9997934be48b142b5996a47927", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/query_response")
async def query_response(message: str, data: str):
    vector, explanation = get_response(message)
    return [vector, explanation]


def generate_response(prompt: str) -> str:
    """ 使用 OpenAI API 生成模型响应 """
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content

def query_vector(message: str, ego_position: str, surrounding_positions: str) -> str:
    """ 生成向量 """
    prompt = vector_template.format(question=message,
                                    ego_position=ego_position,
                                    surrounding_positions=surrounding_positions
                                    )
    return generate_response(prompt)

def analyze_vector(vector: str, message: str, ego_position: str, surrounding_positions: str) -> str:
    """分析向量，包含位置信息"""
    prompt = analysis_template.format(
        vector=vector,
        question=message,
        ego_position=ego_position,
        surrounding_positions=surrounding_positions
    )
    return generate_response(prompt)

def read_positions(file_path: str):
    """从文件中读取位置信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        positions = json.load(file)
    return positions

def get_response(message: str):
    """ 获取向量及其分析 """

    # 从文件中读取自车和周围车辆的位置
    ego_position = read_positions('ego_position.json')  # 自车位置文件
    surrounding_positions = read_positions('surrounding_positions.json')  # 周围车辆位置文件

    # 将位置信息转换为字符串，便于插入到模板中
    ego_position_str = json.dumps(ego_position, ensure_ascii=False)
    surrounding_positions_str = json.dumps(surrounding_positions, ensure_ascii=False)

    vector = query_vector(message)
    explanation = analyze_vector(vector, message)
    return vector, explanation

def query_vector_x(vector):
    # vector = [float(val) for val in vector]  # 转换为浮点数数组
    return vector

# if __name__ == "__main__":
#     message = "我赶时间，想要开的快一些，减少舒适性的考虑"
#     vector, explanation = get_response(message)
#     print("生成的向量:", vector)
#     # print("向量分析:", explanation)
#     uvicorn.run(app, host="127.0.0.1", port=8004, reload=False)

if __name__ == "__main__":
    message = "我赶时间，想要开的快一些，减少舒适性的考虑"
    vector, explanation = get_response(message)
    # 打印生成的向量
    # print(vector)
    vector = query_vector_x(vector)
    print(vector)

    # uvicorn.run(app, host="127.0.0.1", port=8004, reload=False)
