import os
import re
import json
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from openai import OpenAI
from dotenv import load_dotenv

from rag.embedder import LocalEmbedder
from rag.ingest import extract_pdf, extract_docx
from rag.utils import chunk_text
from rag.store import (
    init_db, load_or_create_index, save_index,
    insert_document, list_documents, insert_chunk,
    fetch_chunks_by_faiss_ids, next_faiss_id,
    get_profile, set_profile, get_prefs, set_prefs
)

load_dotenv()

app = Flask(__name__)

# ====== LLM Client (DeepSeek compatible OpenAI SDK) ======
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ====== RAG init ======
init_db()
embedder = LocalEmbedder()
DIM = embedder.dim
index = load_or_create_index(DIM)

# -----------------------------
# RAG config
# -----------------------------
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.18"))  # ✅ 默认更低，命中更稳定
RAG_MAX_CHARS_PER_CHUNK = int(os.getenv("RAG_MAX_CHARS_PER_CHUNK", "700"))
RAG_MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "8"))

STRICT_TRIGGER_PATTERNS = [
    "必须基于", "只根据", "仅根据", "严格基于", "只能依据",
    "不要用常识", "不要扩展", "只用资料", "from the document only"
]

def is_strict_query(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(p.lower() in t for p in STRICT_TRIGGER_PATTERNS)

# -----------------------------
# Identity / Persona
# -----------------------------
ASSISTANT_NAME = "Mercurial"
ASSISTANT_NAME_ZH = "墨丘利"

def build_system_prompt():
    """
    每次请求动态重建系统提示词：
    - 注入用户偏好 + 记忆
    - 强化 RAG 引用规则
    - 固化 Mercurial 人设：优雅、古典、敏捷、智能的指引者
    """
    prefs = get_prefs()
    profile = get_profile()

    lines = [
        f"你是 {ASSISTANT_NAME}（{ASSISTANT_NAME_ZH}）。",
        "灵感源自 Mercury（水银）的流动金属特性，也象征罗马神话中传递信息、速度极快的使者神。",
        "你的风格：优雅、古典、富有智能；表达清晰、克制但有温度；善于把复杂问题讲明白并给出可执行步骤。",
        "你的职责：作为一个专业助手，擅长网络安全、软件工程、RAG/检索增强问答的最佳实践与调试。",
        "",
        f"输出语言偏好: {prefs.get('language', 'zh')}",
        f"语气偏好: {prefs.get('tone', '优雅、专业、简洁')}",
        f"格式偏好: {prefs.get('format_hint', '先结论后分点，最后给 next steps')}",
        f"引用格式要求: {prefs.get('cite_style', '仅在确有引用时输出 [1][2]')}",
        "",
        "RAG 规则：",
        "1) 若提供了【资料片段】（标签形如 [1][2]），你可以引用它们来支撑回答，并在对应句子后标注 [n]。",
        "2) 若没有提供任何【资料片段】，严禁输出形如 [1] 的引用标签（避免假引用）。",
        "3) 默认模式下：资料片段是“辅助证据”，不足时也要给出可用答案，并明确哪些是(推理/常识)。",
        "4) 仅当用户明确要求“必须/仅根据资料”时，才严格限制在资料范围内，不足则直说资料不足。"
    ]

    memory = (profile.get("memory") or "").strip()
    if memory:
        lines.append("")
        lines.append("用户长期信息/偏好补充（仅当相关时使用，避免无关提及）:")
        lines.append(memory)

    return {"role": "system", "content": "\n".join(lines)}

# ====== Chat state ======
conversation_history = [build_system_prompt()]

# ====== Utility: save code blocks ======
def save_code_blocks(text: str):
    code_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)
    blocks = code_pattern.findall(text)
    saved_files = []

    if blocks:
        os.makedirs("codes", exist_ok=True)
        for idx, (lang, content) in enumerate(blocks):
            lang = lang.lower() if lang else "txt"
            ext_map = {
                "python": "py", "py": "py",
                "markdown": "md", "md": "md",
                "javascript": "js", "js": "js",
                "c": "c", "cpp": "cpp",
                "java": "java"
            }
            extension = ext_map.get(lang, lang)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"codes/code_{timestamp}_{idx}.{extension}"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content.strip())
            saved_files.append(filename)
    return saved_files

# ====== RAG context builder ======
def build_rag_context(message: str, top_k: int = 6):
    """
    返回：
      - ctx_lines_used: 注入 prompt 的引用片段（阈值过滤后）
      - citations_used: 与 ctx_lines_used 对应的引用卡片
      - retrieved: 原始 topK 检索到的候选片段（无论阈值，给前端展示/调试）
    """
    if index.ntotal <= 0:
        return [], [], []

    top_k = max(1, min(int(top_k), RAG_MAX_CONTEXT_CHUNKS))

    qv = embedder.embed([message])
    D, I = index.search(qv, top_k)

    raw_pairs = []
    for score, fid in zip(D[0].tolist(), I[0].tolist()):
        fid = int(fid)
        if fid == -1:
            continue
        raw_pairs.append((float(score), fid))

    raw_pairs.sort(key=lambda x: x[0], reverse=True)

    # 1) retrieved：不管阈值，全部拿来给前端展示
    retrieved = []
    if raw_pairs:
        raw_ids = [fid for (_, fid) in raw_pairs]
        raw_ctxs = fetch_chunks_by_faiss_ids(raw_ids)

        for i, (ctx, (score, _fid)) in enumerate(zip(raw_ctxs, raw_pairs), start=1):
            snippet = (ctx["content"] or "").replace("\n", " ").strip()
            if len(snippet) > RAG_MAX_CHARS_PER_CHUNK:
                snippet = snippet[:RAG_MAX_CHARS_PER_CHUNK] + "..."
            retrieved.append({
                "tag": f"[cand{i}]",
                "filename": ctx["filename"],
                "locator": ctx["locator"],
                "snippet": snippet,
                "score": round(score, 3)
            })

    # 2) used：过滤低分，只把足够相关的注入 prompt + citations
    filtered = [(s, fid) for (s, fid) in raw_pairs if s >= RAG_MIN_SCORE]
    if not filtered:
        return [], [], retrieved

    ids_used = [fid for (_, fid) in filtered]
    contexts = fetch_chunks_by_faiss_ids(ids_used)

    citations_used = []
    ctx_lines_used = []
    for i, (ctx, (score, _fid)) in enumerate(zip(contexts, filtered), start=1):
        tag = f"[{i}]"
        snippet = (ctx["content"] or "").replace("\n", " ").strip()
        if len(snippet) > RAG_MAX_CHARS_PER_CHUNK:
            snippet = snippet[:RAG_MAX_CHARS_PER_CHUNK] + "..."

        ctx_lines_used.append(
            f"{tag} 文件: {ctx['filename']} 位置: {ctx['locator']}\n内容: {snippet}"
        )
        citations_used.append({
            "tag": tag,
            "filename": ctx["filename"],
            "locator": ctx["locator"],
            "snippet": snippet,
            "score": round(score, 3)
        })

    return ctx_lines_used, citations_used, retrieved

# ====== Delete docs support ======
DB_PATH = "data/app.db"

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def rebuild_faiss_index_from_db():
    global index

    conn = _db()
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, content
        FROM chunks
        ORDER BY id ASC
    """).fetchall()

    if not rows:
        empty = load_or_create_index(DIM)
        if empty.ntotal != 0:
            empty.reset()
        save_index(empty)
        index = empty
        conn.close()
        return

    chunk_ids = [int(r["id"]) for r in rows]
    texts = [r["content"] for r in rows]

    vecs = embedder.embed(texts)
    new_index = load_or_create_index(DIM)
    new_index.reset()
    new_index.add(vecs)
    save_index(new_index)
    index = new_index

    cur.executemany(
        "UPDATE chunks SET faiss_id = ? WHERE id = ?",
        [(i, chunk_ids[i]) for i in range(len(chunk_ids))]
    )

    conn.commit()
    conn.close()

def delete_document_by_id(doc_id: int):
    conn = _db()
    cur = conn.cursor()

    doc = cur.execute("SELECT id, filename, filepath FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        conn.close()
        return False, "文档不存在"

    filepath = doc["filepath"]

    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass

    rebuild_faiss_index_from_db()
    return True, "删除成功"

# ====== Routes ======
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/docs", methods=["GET"])
def docs_list():
    return jsonify({"documents": list_documents(), "vector_count": int(index.ntotal)})

@app.route("/docs/<int:doc_id>", methods=["DELETE"])
def delete_doc(doc_id: int):
    ok, msg = delete_document_by_id(doc_id)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 404
    return jsonify({"ok": True, "message": msg, "vector_count": int(index.ntotal)})

@app.route("/profile", methods=["GET", "PUT"])
def profile():
    if request.method == "GET":
        return jsonify(get_profile())
    data = request.json or {}
    set_profile(data.get("memory", ""))
    return jsonify({"ok": True})

@app.route("/prefs", methods=["GET", "PUT"])
def prefs():
    if request.method == "GET":
        return jsonify(get_prefs())
    data = request.json or {}
    set_prefs(
        data.get("language", "zh"),
        data.get("tone", ""),
        data.get("format_hint", ""),
        data.get("cite_style", "")
    )
    return jsonify({"ok": True})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "缺少文件"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx"]:
        return jsonify({"error": "仅支持 .pdf 和 .docx"}), 400

    os.makedirs("data/uploads", exist_ok=True)
    save_path = os.path.join("data/uploads", filename)

    if os.path.exists(save_path):
        stem, ext2 = os.path.splitext(filename)
        filename = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext2}"
        save_path = os.path.join("data/uploads", filename)

    f.save(save_path)

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_id = insert_document(filename, save_path, created_at)

    if ext == ".pdf":
        parts = extract_pdf(save_path)
    else:
        parts = extract_docx(save_path)

    all_chunks, all_locators = [], []
    for locator, text in parts:
        for ch in chunk_text(text):
            all_chunks.append(ch)
            all_locators.append(locator)

    if not all_chunks:
        return jsonify({"ok": True, "doc_id": doc_id, "chunks": 0, "filename": filename})

    vecs = embedder.embed(all_chunks)
    start_id = next_faiss_id(index)
    index.add(vecs)
    save_index(index)

    for offset, (locator, content) in enumerate(zip(all_locators, all_chunks)):
        insert_chunk(doc_id, int(locator), content, int(start_id + offset))

    return jsonify({"ok": True, "doc_id": doc_id, "chunks": len(all_chunks), "filename": filename})

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    data = request.json or {}
    user_input = (data.get("message") or "").strip()
    mode = data.get("mode", "chat")
    use_rag = bool(data.get("use_rag", True))
    top_k = int(data.get("top_k", 6))
    rag_strict = bool(data.get("rag_strict", False)) or is_strict_query(user_input)

    if not user_input:
        return jsonify({"error": "消息不能为空"}), 400

    # rebuild system prompt each request
    conversation_history = [build_system_prompt()] + [m for m in conversation_history if m.get("role") != "system"]

    processed_input = user_input
    if mode == "c":
        processed_input = f"【请用 C 语言实现以下需求，并直接给出代码块】：{user_input}"
    elif mode == "python":
        processed_input = f"【请用 Python 语言实现以下需求，并直接给出代码块】：{user_input}"
    elif mode == "java":
        processed_input = f"【请用 Java 语言实现以下需求，并直接给出代码块】：{user_input}"

    rag_lines, citations_used, retrieved = ([], [], [])
    if use_rag:
        rag_lines, citations_used, retrieved = build_rag_context(user_input, top_k=top_k)

    if rag_lines:
        if rag_strict:
            processed_input = (
                processed_input
                + "\n\n【资料片段（仅可使用这些内容作答）】\n"
                + "\n\n".join(rag_lines)
                + "\n\n要求：只能依据资料片段回答；资料不足时请明确说“资料不足”，不要用常识补全。"
            )
        else:
            processed_input = (
                processed_input
                + "\n\n【资料片段（可作为辅助证据）】\n"
                + "\n\n".join(rag_lines)
                + "\n\n要求：优先给出可执行、可用的答案；能被资料支撑的部分请引用对应标签 [1][2]；"
                  "资料未覆盖的部分请明确标注“(推理/常识)”或“(资料未覆盖)”。不要因为资料不全就拒答。"
            )
    else:
        processed_input = (
            processed_input
            + "\n\n注意：本次没有提供任何资料片段，因此不要输出形如 [1] 的引用标签。"
            + "\n\n要求：请正常回答并尽量给出可执行步骤；如果需要更多信息才能更准确，请提出1-3个关键澄清问题。"
        )

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
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            conversation_history.append({"role": "assistant", "content": full_answer})

            trigger_save = (mode in ["c", "python", "java"]) or user_input.lower().startswith("code")
            files = save_code_blocks(full_answer) if trigger_save else []

            meta = {
                "done": True,
                "files": files,
                "citations": citations_used,
                "retrieved": retrieved,   # ✅ 候选片段：保证前端“总有卡片可展示”
                "rag": {
                    "enabled": bool(use_rag),
                    "strict": bool(rag_strict),
                    "min_score": RAG_MIN_SCORE,
                    "hits_used": len(citations_used),
                    "hits_retrieved": len(retrieved),
                    "index_total": int(index.ntotal)
                }
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    # ✅ 清空后也用 Mercurial 的 system prompt（含偏好与记忆）
    conversation_history = [build_system_prompt()]
    return jsonify({"status": "success", "message": "对话历史已清空"})

@app.route("/quit", methods=["POST"])
def quit_server():
    os._exit(0)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
