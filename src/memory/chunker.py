import re
from dataclasses import dataclass
from src.memory.constant import CHUNK_THRESHOLD, CHUNK_MAX_CHARS, CHUNK_OVERLAP


@dataclass
class Chunk:
    content: str
    chunk_type: str  # "whole" | "text" | "code" | "mixed"
    chunk_index: int


class DialogChunker:
    """将助手回答按自然语言/代码块边界切片。

    短回答（< CHUNK_THRESHOLD）不切片，返回单块 type="whole"。
    长回答先按代码围栏分割，代码块保持完整，自然语言按段落组块。
    """

    def chunk(self, text: str) -> list[Chunk]:
        if len(text) < CHUNK_THRESHOLD:
            return [Chunk(content=text, chunk_type="whole", chunk_index=0)]

        segments = self._split_code_fences(text)

        chunks: list[Chunk] = []
        idx = 0
        text_buf: list[str] = []
        text_chars = 0

        for seg_type, seg_text in segments:
            if seg_type == "code":
                if text_buf:
                    for c in self._chunk_text("\n".join(text_buf), idx):
                        chunks.append(c)
                        idx += 1
                    text_buf = []
                    text_chars = 0
                chunks.append(Chunk(content=seg_text, chunk_type="code", chunk_index=idx))
                idx += 1
            else:
                text_buf.append(seg_text)
                text_chars += len(seg_text)
                if text_chars >= CHUNK_MAX_CHARS:
                    for c in self._chunk_text("\n".join(text_buf), idx):
                        chunks.append(c)
                        idx += 1
                    text_buf = []
                    text_chars = 0

        if text_buf:
            for c in self._chunk_text("\n".join(text_buf), idx):
                chunks.append(c)

        if len(chunks) == 1:
            chunks[0].chunk_type = "whole"

        return chunks

    def _split_code_fences(self, text: str) -> list[tuple[str, str]]:
        """按 markdown 代码围栏分割，返回 [(type, text), ...]。"""
        pattern = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)
        parts = []
        last = 0
        for m in pattern.finditer(text):
            if m.start() > last:
                parts.append(("text", text[last:m.start()]))
            parts.append(("code", m.group(1)))
            last = m.end()
        if last < len(text):
            parts.append(("text", text[last:]))
        return parts if parts else [("text", text)]

    def _chunk_text(self, text: str, start_idx: int) -> list[Chunk]:
        """将自然语言文本按段落组块，重叠 OVERLAP 字符。"""
        if len(text) < CHUNK_MAX_CHARS:
            return [Chunk(content=text, chunk_type="text", chunk_index=start_idx)]

        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current: list[str] = []
        chars = 0

        for para in paragraphs:
            if chars + len(para) > self.MAX_CHARS and current:
                chunks.append(Chunk(content="\n\n".join(current), chunk_type="text", chunk_index=start_idx + len(chunks)))
                overlap_paras: list[str] = []
                overlap_chars = 0
                for p in reversed(current):
                    if overlap_chars + len(p) > CHUNK_OVERLAP:
                        break
                    overlap_paras.insert(0, p)
                    overlap_chars += len(p)
                current = overlap_paras
                chars = overlap_chars

            current.append(para)
            chars += len(para) + 2

        if current:
            chunks.append(Chunk(content="\n\n".join(current), chunk_type="text", chunk_index=start_idx + len(chunks)))

        return chunks