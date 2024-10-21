import subprocess
import time

# 目标Python脚本的路径  记得修改
target_script = "openai_interaction.py"
conda_env_name = "chatgpt"


def run_python_script_in_conda(script_path, env_name):
    try:
        print(f"正在使用 Conda 环境 {env_name} 运行 Python 脚本：{script_path}")
        cmd = 'conda run -n chatgpt & python /home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/openai_interaction.py'
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            print("Python 脚本执行成功。")
            # 获取并输出标准输出中的向量
            x = result.stdout.strip()  # 去掉输出中的换行符等
            print(f"从脚本中捕获到的向量 x: {x}")
        else:
            print(f"Python 脚本执行失败，返回码：{result.returncode}")
            print(f"错误信息：{result.stderr}")
    except Exception as e:
        print(f"运行 Python 脚本时发生异常：{e}")

def monitor_script():
    # while True:
        run_python_script_in_conda(target_script, conda_env_name)
        # # 每次运行完成后暂停20秒
        print("chatgpt场景理解完毕...")
        # print("等待20秒后重新运行脚本...")
        # time.sleep(20)
#
# 启动监控
monitor_script()
