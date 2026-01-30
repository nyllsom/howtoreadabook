def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple character-based chunker for demo."""
    text = (text or "").replace("\u0000", "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks
