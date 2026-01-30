import os
import sqlite3
from typing import Any
import faiss

DB_PATH = "data/app.db"
INDEX_PATH = "data/faiss.index"

def ensure_dirs():
    os.makedirs("data/uploads", exist_ok=True)

def connect():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        locator INTEGER NOT NULL,
        content TEXT NOT NULL,
        faiss_id INTEGER NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profile(
        id INTEGER PRIMARY KEY CHECK (id = 1),
        memory TEXT NOT NULL DEFAULT ''
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_prefs(
        id INTEGER PRIMARY KEY CHECK (id = 1),
        language TEXT NOT NULL DEFAULT 'zh',
        tone TEXT NOT NULL DEFAULT '专业但可爱',
        format_hint TEXT NOT NULL DEFAULT '先给结论，再分点说明，必要时给下一步',
        cite_style TEXT NOT NULL DEFAULT '在回答末尾输出引用标签，如：[1][2]'
    )
    """)
    # seed singleton rows
    cur.execute("INSERT OR IGNORE INTO user_profile(id, memory) VALUES(1, '')")
    cur.execute("INSERT OR IGNORE INTO user_prefs(id, language, tone, format_hint, cite_style) VALUES(1, 'zh', '专业但可爱', '先给结论，再分点说明，必要时给下一步', '在回答末尾输出引用标签，如：[1][2]')")
    conn.commit()
    conn.close()

def load_or_create_index(dim: int):
    ensure_dirs()
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    index = faiss.IndexFlatIP(dim)  # cosine if embeddings are normalized
    faiss.write_index(index, INDEX_PATH)
    return index

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def insert_document(filename: str, filepath: str, created_at: str) -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO documents(filename, filepath, created_at) VALUES(?,?,?)", (filename, filepath, created_at))
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return int(doc_id)

def list_documents() -> list[dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, filename, created_at FROM documents ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_chunk(document_id: int, locator: int, content: str, faiss_id: int):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chunks(document_id, locator, content, faiss_id) VALUES(?,?,?,?)",
        (document_id, locator, content, faiss_id)
    )
    conn.commit()
    conn.close()

def fetch_chunks_by_faiss_ids(ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute(f"""
        SELECT c.faiss_id, c.content, c.locator, d.filename
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.faiss_id IN ({placeholders})
    """, ids).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    order = {fid: i for i, fid in enumerate(ids)}
    data.sort(key=lambda x: order.get(x["faiss_id"], 10**9))
    return data

def next_faiss_id(index) -> int:
    return int(index.ntotal)

def get_profile() -> dict[str, Any]:
    conn = connect()
    cur = conn.cursor()
    row = cur.execute("SELECT memory FROM user_profile WHERE id=1").fetchone()
    conn.close()
    return {"memory": row["memory"] if row else ""}

def set_profile(memory: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute("UPDATE user_profile SET memory=? WHERE id=1", (memory or "",))
    conn.commit()
    conn.close()

def get_prefs() -> dict[str, Any]:
    conn = connect()
    cur = conn.cursor()
    row = cur.execute("SELECT language, tone, format_hint, cite_style FROM user_prefs WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {"language": "zh", "tone": "专业但可爱", "format_hint": "", "cite_style": ""}
    return dict(row)

def set_prefs(language: str, tone: str, format_hint: str, cite_style: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "UPDATE user_prefs SET language=?, tone=?, format_hint=?, cite_style=? WHERE id=1",
        (language or "zh", tone or "", format_hint or "", cite_style or "")
    )
    conn.commit()
    conn.close()
