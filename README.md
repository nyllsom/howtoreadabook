# CyberCat AI - Vibe Coding Edition 🐾

CyberCat AI 是一个基于 DeepSeek 模型开发的流式 AI 助手，专为网络安全和快速编程（Vibe Coding）设计。它不仅具备专业的网络安全猫娘人格，还能根据预设模式（Python、C、Java）自动提取并保存生成的代码块到本地。

## 🌟 核心特性

-   **流式输出 (Streaming)**：采用 SSE (Server-Sent Events) 技术，实现像 ChatGPT 一样的打字机实时回复效果。
-   **自动化代码提取**：在编程模式下，AI 生成的代码块会自动解析并保存到本地 `codes/` 文件夹中。
-   **多模式切换**：
    -   💬 **普通对话**：日常安全咨询与闲聊。
    -   📁 **C 语言模式**：专注 C 语言实现，自动保存 `.c` 文件。
    -   🐍 **Python 模式**：专注 Python 脚本编写，自动保存 `.py` 文件。
    -   ☕ **Java 模式**：专注 Java 逻辑开发，自动保存 `.java` 文件。
-   **精美 UI**：基于 Tailwind CSS 构建的深色系网络安全风格界面，支持 Markdown 渲染和语法高亮。
-   **上下文记忆**：支持连续对话，并提供一键清除历史功能。

## 📁 项目结构

```text
.
├── .env                # 配置文件（存放 API Key）
├── app.py              # Flask 后端核心逻辑
├── codes/              # 自动生成的代码存放目录（Git 已忽略）
├── templates/
│   └── index.html      # 前端界面
└── README.md           # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备
确保您的系统中已安装 Python 3.8+。

### 2. 安装依赖
打开终端，运行以下命令安装必要的库：
```bash
pip install flask openai python-dotenv
```

### 3. 配置 API Key
在项目根目录下创建一个 `.env` 文件，内容如下：
```env
DEEPSEEK_API_KEY=您的DEEPSEEK_API_KEY
```

### 4. 运行应用
执行以下命令启动服务器：
```bash
python app.py
```
启动后，在浏览器访问 `http://127.0.0.1:5000` 即可开始使用。

## 🛠 使用技巧

-   **代码自动保存**：切换到对应的编程模式（如 Python），输入需求后，生成的代码会按时间戳命名存放在 `codes/` 目录下。
-   **强制保存**：即使在普通对话模式下，只要输入的消息以 `code` 开头，助手也会尝试提取并保存代码块。
-   **清理缓存**：点击“Clear”按钮可以清空当前的对话上下文，开始新的话题。

## ⚠️ 注意事项

-   本程序会自动在本地创建文件，请确保程序具有对应目录的读写权限。
-   请妥善保管 `.env` 文件，切勿将其上传到公开的代码仓库。

---
*Developed with ❤️ for Cyber Security Enthusiasts.*
