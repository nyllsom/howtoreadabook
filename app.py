import os
import re
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = {"role": "system", "content": "你是一个专业的网络安全猫娘。"}
conversation_history = [SYSTEM_PROMPT]

def save_code_blocks(text):
    code_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)
    blocks = code_pattern.findall(text)
    saved_files = []
    
    if blocks:
        if not os.path.exists("codes"):
            os.makedirs("codes")
        for idx, (lang, content) in enumerate(blocks):
            lang = lang.lower() if lang else "txt"
            ext_map = {
                "python": "py", "py": "py", 
                "markdown": "md", "md": "md", 
                "javascript": "js", "js": "js",
                "c": "c", "cpp": "cpp",
                "java": "java"  # 新增 Java 映射
            }
            extension = ext_map.get(lang, lang)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"codes/code_{timestamp}_{idx}.{extension}"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content.strip())
            saved_files.append(filename)
    return saved_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.json
    user_input = data.get('message', '')
    mode = data.get('mode', 'chat')
    
    if not user_input:
        return jsonify({"error": "消息不能为空"}), 400

    processed_input = user_input
    if mode == 'c':
        processed_input = f"【请用 C 语言实现以下需求，并直接给出代码块】：{user_input}"
    elif mode == 'python':
        processed_input = f"【请用 Python 语言实现以下需求，并直接给出代码块】：{user_input}"
    elif mode == 'java': # 新增 Java 模式 Prompt 处理
        processed_input = f"【请用 Java 语言实现以下需求，并直接给出代码块】：{user_input}"

    conversation_history.append({"role": "user", "content": processed_input})

    def generate():
        full_answer = ""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=conversation_history,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            conversation_history.append({"role": "assistant", "content": full_answer})
            
            # 增加 'java' 到触发保存的条件中
            trigger_save = (mode in ['c', 'python', 'java']) or user_input.lower().startswith("code")
            files = []
            if trigger_save:
                files = save_code_blocks(full_answer)
            
            yield f"data: {json.dumps({'done': True, 'files': files})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/clear', methods=['POST'])
def clear():
    global conversation_history
    conversation_history = [SYSTEM_PROMPT]
    return jsonify({"status": "success", "message": "对话历史已清空"})

@app.route('/quit', methods=['POST'])
def quit_server():
    os._exit(0)

if __name__ == '__main__':
    app.run(debug=True, port=5000)