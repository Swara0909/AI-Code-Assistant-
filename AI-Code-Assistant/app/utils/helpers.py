"""
Document loading and preprocessing helpers.
Supports: .py .js .ts .java .cpp .c .cs .go .rs .html .css .md .txt .pdf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
}

_TEXT_EXTENSIONS = {".txt", ".md", ".html", ".css", ".json", ".yaml", ".yml"}


def _get_splitter(ext: str) -> RecursiveCharacterTextSplitter:
    lang_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS, ".jsx": Language.JS,
        ".ts": Language.JS, ".tsx": Language.JS,
        ".java": Language.JAVA,
        ".cpp": Language.CPP, ".c": Language.CPP,
        ".go": Language.GO,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".html": Language.HTML, ".htm": Language.HTML,
    }
    if ext in lang_map:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_map[ext],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def load_file(file_path: Union[str, Path]) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

    raw_docs = loader.load()
    for doc in raw_docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["file_name"] = path.name
        doc.metadata["extension"] = ext

    return _get_splitter(ext).split_documents(raw_docs)


def load_directory(dir_path: Union[str, Path]) -> List[Document]:
    supported = _CODE_EXTENSIONS | _TEXT_EXTENSIONS | {".pdf"}
    all_docs: List[Document] = []

    for root, _, files in os.walk(dir_path):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in supported:
                try:
                    all_docs.extend(load_file(fpath))
                except Exception as exc:
                    print(f"[warn] Could not load {fpath}: {exc}")

    return all_docs


def load_text_snippet(text: str, source: str = "user_input") -> List[Document]:
    """Wrap a raw text/code string as chunked Documents."""
    doc = Document(page_content=text, metadata={"source": source})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents([doc])
