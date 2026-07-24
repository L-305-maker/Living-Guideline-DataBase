"""SQLite metadata store with FTS5 BM25 retrieval."""

from __future__ import annotations

from contextlib import closing

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from src.retrieval.common import (
    enrich_chunk_metadata as helper_enrich_chunk_metadata,
    parse_time_range as helper_time_range,
)
from src.retrieval.chunk_normalizer import iter_normalized_chunks
from src.retrieval.rrf import rrf_fusion
from src.utils.io import DATA_DIR, ensure_parent, read_jsonl
from src.utils.records import (
    department_text as helper_department_text,
    publication_year as helper_year,
    truthy as helper_truthy,
)


DEFAULT_DB_PATH = DATA_DIR / "index" / "rag.sqlite"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?|[\u4e00-\u9fff]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MEDICAL_TERMS = (
    "糖尿病",
    "胰岛素",
    "胰岛素泵",
    "高血压",
    "冠心病",
    "心力衰竭",
    "慢性阻塞性肺疾病",
    "哮喘",
    "感染",
    "指南",
    "共识",
    "诊断",
    "治疗",
    "管理",
    "筛查",
    "预防",
    "并发症",
    "妊娠",
    "儿童",
    "成人",
    "老年",
    "肿瘤",
    "肾病",
    "卒中",
    "抗菌药物",
    "风险评估",
)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    document_kind TEXT NOT NULL DEFAULT 'guideline',
    source_file TEXT NOT NULL DEFAULT '',
    markdown_raw_path TEXT NOT NULL DEFAULT '',
    markdown_clean_path TEXT NOT NULL DEFAULT '',
    content_sha256 TEXT NOT NULL DEFAULT '',
    cleaning_quality TEXT NOT NULL DEFAULT '',
    cleaning_flags TEXT NOT NULL DEFAULT '',
    source_pdf_text_quality TEXT NOT NULL DEFAULT '',
    source_pdf_needs_ocr INTEGER NOT NULL DEFAULT 0,
    source_pdf_is_scanned INTEGER NOT NULL DEFAULT 0,
    pdf_text_quality TEXT NOT NULL DEFAULT '',
    pdf_needs_ocr INTEGER NOT NULL DEFAULT 0,
    pdf_is_scanned INTEGER NOT NULL DEFAULT 0,
    ocr_engine TEXT NOT NULL DEFAULT '',
    ocr_applied INTEGER NOT NULL DEFAULT 0,
    ocr_status TEXT NOT NULL DEFAULT '',
    ocr_error TEXT NOT NULL DEFAULT '',
    content_md TEXT NOT NULL DEFAULT '',
    title_zh_tokens TEXT NOT NULL DEFAULT '',
    abstract_zh_tokens TEXT NOT NULL DEFAULT '',
    content_zh_tokens TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sections (
    section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path TEXT NOT NULL DEFAULT '[]',
    section_path_text TEXT NOT NULL DEFAULT '',
    heading TEXT,
    heading_level INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    is_reference_section INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL DEFAULT '',
    title_zh_tokens TEXT NOT NULL DEFAULT '',
    section_path_zh_tokens TEXT NOT NULL DEFAULT '',
    content_zh_tokens TEXT NOT NULL DEFAULT '',
    UNIQUE (doc_id, section_index),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    section_path TEXT NOT NULL DEFAULT '[]',
    section_path_text TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    retrieval_text TEXT NOT NULL DEFAULT '',
    chunk_type TEXT NOT NULL DEFAULT 'other',
    token_count INTEGER NOT NULL DEFAULT 0,
    retrieval_key TEXT NOT NULL UNIQUE,
    is_background INTEGER NOT NULL DEFAULT 0,
    is_reference_section INTEGER NOT NULL DEFAULT 0,
    title_zh_tokens TEXT NOT NULL DEFAULT '',
    section_path_zh_tokens TEXT NOT NULL DEFAULT '',
    content_zh_tokens TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_cards (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    card_text TEXT NOT NULL DEFAULT '',
    title_zh_tokens TEXT NOT NULL DEFAULT '',
    card_zh_tokens TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS document_views (
    view_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    view_type TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 1.0,
    title TEXT NOT NULL,
    publication_date TEXT NOT NULL DEFAULT 'unknown',
    publication_year INTEGER,
    source_institution TEXT NOT NULL DEFAULT 'Unknown',
    clinical_department TEXT NOT NULL DEFAULT '未分类',
    text TEXT NOT NULL DEFAULT '',
    title_zh_tokens TEXT NOT NULL DEFAULT '',
    text_zh_tokens TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    section_path_text,
    clinical_department,
    content,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_zh_fts USING fts5(
    title_zh_tokens,
    section_path_zh_tokens,
    content_zh_tokens,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE VIRTUAL TABLE IF NOT EXISTS document_cards_fts USING fts5(
    title,
    clinical_department,
    card_text,
    content='document_cards',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE VIRTUAL TABLE IF NOT EXISTS document_cards_zh_fts USING fts5(
    title_zh_tokens,
    card_zh_tokens,
    content='document_cards',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE VIRTUAL TABLE IF NOT EXISTS document_views_fts USING fts5(
    view_type,
    title,
    clinical_department,
    text,
    content='document_views',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE VIRTUAL TABLE IF NOT EXISTS document_views_zh_fts USING fts5(
    view_type,
    title_zh_tokens,
    text_zh_tokens,
    content='document_views',
    content_rowid='rowid',
    tokenize='unicode61',
    detail='column'
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_institution);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_kind_department ON documents(document_kind, clinical_department);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(publication_date);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(publication_year);
CREATE INDEX IF NOT EXISTS idx_documents_pdf_quality ON documents(pdf_text_quality);
CREATE INDEX IF NOT EXISTS idx_documents_pdf_needs_ocr ON documents(pdf_needs_ocr);
CREATE INDEX IF NOT EXISTS idx_documents_ocr_status ON documents(ocr_status);
CREATE INDEX IF NOT EXISTS idx_documents_cleaning_quality ON documents(cleaning_quality);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(doc_id);
CREATE INDEX IF NOT EXISTS idx_sections_department ON sections(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_institution);
CREATE INDEX IF NOT EXISTS idx_chunks_department ON chunks(clinical_department);
CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(publication_date);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(publication_year);
CREATE INDEX IF NOT EXISTS idx_chunks_reference ON chunks(is_reference_section);
CREATE INDEX IF NOT EXISTS idx_document_cards_source ON document_cards(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_cards_department ON document_cards(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_cards_date ON document_cards(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_views_doc ON document_views(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_views_type ON document_views(view_type);
CREATE INDEX IF NOT EXISTS idx_document_views_source ON document_views(source_institution);
CREATE INDEX IF NOT EXISTS idx_document_views_department ON document_views(clinical_department);
CREATE INDEX IF NOT EXISTS idx_document_views_date ON document_views(publication_date);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn




def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))




def helper_section_path_text(section_path: Iterable[Any]) -> str:
    return " ".join(str(item) for item in section_path if item is not None)




def helper_zh_token_text(text: str, max_chars: int | None = None) -> str:
    # 中文检索同时保留医学词和有限二三元组；去重与长度上限用于控制索引膨胀。
    if max_chars is not None:
        text = (text or "")[:max_chars]
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().lower()
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    for part in TOKEN_RE.findall(text or ""):
        if CJK_RE.search(part):
            if len(part) <= 16:
                add(part)
            for term in MEDICAL_TERMS:
                if term in part:
                    add(term)
            for n in (2, 3):
                if len(part) >= n:
                    for index in range(0, len(part) - n + 1):
                        add(part[index : index + n])
        else:
            add(part)
    return " ".join(tokens)


def helper_fts_query(query: str) -> str:
    tokens = [token.replace('"', '""') for token in TOKEN_RE.findall(query or "")]
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens)


def helper_zh_fts_query(query: str) -> str:
    # 查询词按医学词和有限 n-gram 展开，最终数量受限以避免生成过大的 FTS 表达式。
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip().lower()
        if token and token not in seen:
            seen.add(token)
            tokens.append(token.replace('"', '""'))

    for part in TOKEN_RE.findall(query or ""):
        if CJK_RE.search(part):
            for term in MEDICAL_TERMS:
                if term in part:
                    add(term)
            if len(part) <= 3:
                add(part)
            for n in (2, 3):
                if len(part) >= n:
                    for index in range(0, len(part) - n + 1):
                        add(part[index : index + n])
        else:
            add(part)
    if not tokens:
        return '""'
    return " OR ".join(f'"{token}"' for token in tokens[:16])




def init_schema(db_path: str | Path = DEFAULT_DB_PATH, reset: bool = False) -> dict[str, Any]:
    path = Path(db_path)
    ensure_parent(path)
    if reset and path.exists():
        path.unlink()
        wal_path = path.with_name(path.name + "-wal")
        shm_path = path.with_name(path.name + "-shm")
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()
    conn = connect(path)
    try:
        document_columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
        if document_columns and "document_kind" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN document_kind TEXT NOT NULL DEFAULT 'guideline'")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "db_path": str(path)}


def helper_clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM document_views_zh_fts")
    conn.execute("DELETE FROM document_views_fts")
    conn.execute("DELETE FROM document_cards_zh_fts")
    conn.execute("DELETE FROM document_cards_fts")
    conn.execute("DELETE FROM chunks_zh_fts")
    conn.execute("DELETE FROM chunks_fts")
    conn.execute("DELETE FROM document_views")
    conn.execute("DELETE FROM document_cards")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM sections")
    conn.execute("DELETE FROM documents")




def helper_department_metadata(value: str) -> dict[str, Any]:
    labels = [label for label in str(value or "").split("|") if label] or ["未分类"]
    return {
        "clinical_department": labels[0],
        "clinical_departments": labels,
        "department_scope": "compositive" if len(labels) > 1 else "single",
    }


def helper_insert_documents(conn: sqlite3.Connection, data_dir: Path) -> int:
    # 主表与 FTS 派生字段必须在同一批次保持一致，任何字段规范化都在写入前完成。
    rows = []
    for rec in read_jsonl(data_dir / "documents.jsonl"):
        clean_path = Path(rec.get("markdown_clean_path") or "")
        content_for_tokens = clean_path.read_text(encoding="utf-8", errors="replace") if clean_path.exists() else ""
        row = (
            rec["doc_id"],
            rec.get("title") or "",
            rec.get("abstract") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            rec.get("document_kind") or "guideline",
            rec.get("source_file") or "",
            rec.get("markdown_raw_path") or "",
            rec.get("markdown_clean_path") or "",
            rec.get("content_sha256") or "",
            rec.get("cleaning_quality") or "",
            rec.get("cleaning_flags") or "",
            rec.get("source_pdf_text_quality") or "",
            1 if helper_truthy(rec.get("source_pdf_needs_ocr")) else 0,
            1 if helper_truthy(rec.get("source_pdf_is_scanned")) else 0,
            rec.get("pdf_text_quality") or "",
            1 if helper_truthy(rec.get("pdf_needs_ocr")) else 0,
            1 if helper_truthy(rec.get("pdf_is_scanned")) else 0,
            rec.get("ocr_engine") or "",
            1 if helper_truthy(rec.get("ocr_applied")) else 0,
            rec.get("ocr_status") or "",
            rec.get("ocr_error") or "",
            content_for_tokens,
            helper_zh_token_text(rec.get("title") or ""),
            helper_zh_token_text(rec.get("abstract") or ""),
            helper_zh_token_text(content_for_tokens, max_chars=8000),
        )
        rows.append(row)
    conn.executemany(
        """
        INSERT INTO documents
        (doc_id, title, abstract, publication_date, publication_year, source_institution, clinical_department, document_kind, source_file,
         markdown_raw_path, markdown_clean_path, content_sha256, cleaning_quality, cleaning_flags,
         source_pdf_text_quality, source_pdf_needs_ocr,
         source_pdf_is_scanned, pdf_text_quality, pdf_needs_ocr, pdf_is_scanned,
         ocr_engine, ocr_applied, ocr_status, ocr_error, content_md, title_zh_tokens,
         abstract_zh_tokens, content_zh_tokens)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    return len(rows)


def helper_document_ids(data_dir: Path) -> set[str]:
    return {record["doc_id"] for record in read_jsonl(data_dir / "documents.jsonl")}


def helper_insert_sections(conn: sqlite3.Connection, data_dir: Path, batch_size: int, allowed_doc_ids: set[str]) -> int:
    # 章节顺序和字符区间来自编码阶段，写库时只做类型规范化，不重新推断结构。
    count = 0
    batch = []
    for path in sorted((data_dir / "sections").glob("*.jsonl")):
        for section_index, rec in enumerate(read_jsonl(path)):
            if rec.get("doc_id") not in allowed_doc_ids:
                continue
            section_path = rec.get("section_path") or []
            section_path_text = helper_section_path_text(section_path)
            batch.append(
                (
                    rec["doc_id"],
                    section_index,
                    rec.get("title") or "",
                    rec.get("publication_date") or "unknown",
                    helper_year(rec.get("publication_date")),
                    rec.get("source_institution") or "Unknown",
                    helper_department_text(rec),
                    serialize_json(section_path),
                    section_path_text,
                    rec.get("heading"),
                    rec.get("heading_level"),
                    rec.get("char_start"),
                    rec.get("char_end"),
                    1 if rec.get("is_reference_section") else 0,
                    rec.get("content") or "",
                    helper_zh_token_text(rec.get("title") or ""),
                    helper_zh_token_text(section_path_text),
                    helper_zh_token_text(rec.get("content") or "", max_chars=8000),
                )
            )
            if len(batch) >= batch_size:
                conn.executemany(
                    """
                    INSERT INTO sections
                    (doc_id, section_index, title, publication_date, publication_year, source_institution, clinical_department,
                     section_path, section_path_text, heading, heading_level, char_start, char_end,
                     is_reference_section, content, title_zh_tokens, section_path_zh_tokens, content_zh_tokens)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    batch,
                )
                count += len(batch)
                batch.clear()
    if batch:
        conn.executemany(
            """
            INSERT INTO sections
            (doc_id, section_index, title, publication_date, publication_year, source_institution, clinical_department,
             section_path, section_path_text, heading, heading_level, char_start, char_end,
             is_reference_section, content, title_zh_tokens, section_path_zh_tokens, content_zh_tokens)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            batch,
        )
        count += len(batch)
    return count


def helper_insert_chunks(conn: sqlite3.Connection, data_dir: Path, batch_size: int) -> int:
    # 分块正文、检索文本和中文分词文本必须逐条对应，避免 FTS 行与主表错位。
    count = 0
    rows = []
    for rec in iter_normalized_chunks(data_dir):
        section_path = rec.get("section_path") or []
        section_path_text = helper_section_path_text(section_path)
        content = rec.get("content") or ""
        retrieval_text = rec.get("retrieval_text") or content
        chunk_type = rec.get("chunk_type") or "other"
        row = (
            rec["chunk_id"],
            rec["doc_id"],
            rec.get("title") or "",
            rec.get("publication_date") or "unknown",
            helper_year(rec.get("publication_date")),
            rec.get("source_institution") or "Unknown",
            helper_department_text(rec),
            serialize_json(section_path),
            section_path_text,
            rec.get("chunk_index") or 0,
            content,
            retrieval_text,
            chunk_type,
            int(rec.get("token_count") or 0),
            rec.get("retrieval_key") or "",
            1 if rec.get("is_background") else 0,
            1 if rec.get("is_reference_section") else 0,
            helper_zh_token_text(rec.get("title") or ""),
            helper_zh_token_text(section_path_text),
            helper_zh_token_text(retrieval_text, max_chars=8000),
        )
        rows.append(row)
        if len(rows) >= batch_size:
            conn.executemany(
                """
                INSERT INTO chunks
                (chunk_id, doc_id, title, publication_date, publication_year, source_institution, clinical_department,
                 section_path, section_path_text, chunk_index, content, retrieval_text, chunk_type, token_count,
                 retrieval_key, is_background, is_reference_section,
                 title_zh_tokens, section_path_zh_tokens,
                 content_zh_tokens)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            count += len(rows)
            rows.clear()
    if rows:
        conn.executemany(
            """
            INSERT INTO chunks
            (chunk_id, doc_id, title, publication_date, publication_year, source_institution, clinical_department,
             section_path, section_path_text, chunk_index, content, retrieval_text, chunk_type, token_count,
             retrieval_key, is_background, is_reference_section,
             title_zh_tokens, section_path_zh_tokens,
             content_zh_tokens)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        count += len(rows)
    return count


def helper_insert_document_cards(conn: sqlite3.Connection, data_dir: Path, allowed_doc_ids: set[str]) -> int:
    path = data_dir / "document_cards.jsonl"
    if not path.exists():
        return 0
    rows = []
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if doc_id not in allowed_doc_ids:
            continue
        card_text = rec.get("card_text") or ""
        rows.append(
            (
                doc_id,
                rec.get("title") or "",
                rec.get("publication_date") or "unknown",
                helper_year(rec.get("publication_date")),
                rec.get("source_institution") or "Unknown",
                helper_department_text(rec),
                card_text,
                helper_zh_token_text(rec.get("title") or ""),
                helper_zh_token_text(card_text, max_chars=12000),
            )
        )
    if rows:
        conn.executemany(
            """
            INSERT INTO document_cards
            (doc_id, title, publication_date, publication_year, source_institution, clinical_department,
             card_text, title_zh_tokens, card_zh_tokens)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
    return len(rows)

def helper_insert_document_views(conn: sqlite3.Connection, data_dir: Path, allowed_doc_ids: set[str], batch_size: int) -> int:
    # 多视图按 view_id 独立写入，同一文档的不同视图不能在入库时互相覆盖。
    path = data_dir / "document_views.jsonl"
    if not path.exists():
        return 0
    count = 0
    rows = []
    insert_sql = """
        INSERT INTO document_views
        (view_id, doc_id, view_type, priority, title, publication_date, publication_year,
         source_institution, clinical_department, text, title_zh_tokens, text_zh_tokens)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """
    for rec in read_jsonl(path):
        doc_id = rec.get("doc_id")
        if doc_id not in allowed_doc_ids:
            continue
        text = rec.get("text") or ""
        rows.append(
            (
                rec.get("view_id") or f"{doc_id}__{rec.get('view_type')}",
                doc_id,
                rec.get("view_type") or "",
                float(rec.get("priority") or 1.0),
                rec.get("title") or "",
                rec.get("publication_date") or "unknown",
                helper_year(rec.get("publication_date")),
                rec.get("source_institution") or "Unknown",
                helper_department_text(rec),
                text,
                helper_zh_token_text(rec.get("title") or ""),
                helper_zh_token_text(text, max_chars=8000),
            )
        )
        if len(rows) >= batch_size:
            conn.executemany(insert_sql, rows)
            count += len(rows)
            rows.clear()
    if rows:
        conn.executemany(insert_sql, rows)
        count += len(rows)
    return count

def build_sqlite_store(
    data_dir: str | Path = DATA_DIR,
    db_path: str | Path = DEFAULT_DB_PATH,
    batch_size: int = 2000,
    reset: bool = True,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    allowed_doc_ids = helper_document_ids(data_path)
    init_schema(db_path, reset=reset)
    with closing(connect(db_path)) as conn:
        helper_clear(conn)
        documents = helper_insert_documents(conn, data_path)
        conn.commit()
        document_cards = helper_insert_document_cards(conn, data_path, allowed_doc_ids)
        conn.commit()
        document_views = helper_insert_document_views(conn, data_path, allowed_doc_ids, batch_size)
        conn.commit()
        sections = helper_insert_sections(conn, data_path, batch_size, allowed_doc_ids)
        conn.commit()
        chunks = helper_insert_chunks(conn, data_path, batch_size)
        conn.commit()
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO chunks_zh_fts(chunks_zh_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO document_cards_fts(document_cards_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO document_cards_zh_fts(document_cards_zh_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO document_views_fts(document_views_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO document_views_zh_fts(document_views_zh_fts) VALUES('rebuild')")
        conn.commit()
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        conn.execute("INSERT INTO chunks_zh_fts(chunks_zh_fts) VALUES('optimize')")
        conn.execute("INSERT INTO document_cards_fts(document_cards_fts) VALUES('optimize')")
        conn.execute("INSERT INTO document_cards_zh_fts(document_cards_zh_fts) VALUES('optimize')")
        conn.execute("INSERT INTO document_views_fts(document_views_fts) VALUES('optimize')")
        conn.execute("INSERT INTO document_views_zh_fts(document_views_zh_fts) VALUES('optimize')")
        conn.commit()
    return {
        "db_path": str(db_path),
        "documents": documents,
        "document_cards": document_cards,
        "document_views": document_views,
        "sections": sections,
        "chunks": chunks,
    }


def stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    with closing(connect(db_path)) as conn:
        payload: dict[str, Any] = {}
        for table in [
            "documents",
            "document_cards",
            "document_views",
            "sections",
            "chunks",
            "chunks_fts",
            "chunks_zh_fts",
            "document_cards_fts",
            "document_cards_zh_fts",
            "document_views_fts",
            "document_views_zh_fts",
        ]:
            payload[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        payload["duplicate_doc_ids"] = conn.execute(
            "SELECT count(*) FROM (SELECT doc_id FROM documents GROUP BY doc_id HAVING count(*) > 1)"
        ).fetchone()[0]
        payload["duplicate_chunk_ids"] = conn.execute(
            "SELECT count(*) FROM (SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING count(*) > 1)"
        ).fetchone()[0]
        payload["unknown_publication_date_documents"] = conn.execute(
            "SELECT count(*) FROM documents WHERE publication_date='unknown'"
        ).fetchone()[0]
        payload["reference_chunks"] = conn.execute("SELECT count(*) FROM chunks WHERE is_reference_section=1").fetchone()[0]
    return payload


def helper_metadata_where(
    alias: str,
    source_institution: str | None,
    clinical_department: str | None,
    time_range: str | dict[str, str] | None,
    params: list[Any],
    document_kind: str | None = None,
) -> str:
    clauses = []
    if source_institution:
        clauses.append(f"{alias}.source_institution LIKE ?")
        params.append(f"%{source_institution}%")
    if clinical_department:
        clauses.append(f"{alias}.clinical_department LIKE ?")
        params.append(f"%{clinical_department}%")
    if document_kind:
        clauses.append(f"EXISTS (SELECT 1 FROM documents d_kind WHERE d_kind.doc_id={alias}.doc_id AND d_kind.document_kind=?)")
        params.append(document_kind)
    start, end = helper_time_range(time_range)
    if start:
        clauses.append(f"{alias}.publication_date >= ?")
        params.append(start)
    if end:
        clauses.append(f"{alias}.publication_date <= ?")
        params.append(end)
    return " AND " + " AND ".join(clauses) if clauses else ""


def helper_search_document_cards_fts(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 50,
    zh: bool = False,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 先由 FTS 产生候选和名次，再补充主表元数据，避免在全文检索前扫描全部文档。
    table = "document_cards_zh_fts" if zh else "document_cards_fts"
    weights = "5.0, 3.0" if zh else "5.0, 1.0, 3.0"
    fts = helper_zh_fts_query(query) if zh else helper_fts_query(query)
    params: list[Any] = [fts]
    metadata_sql = helper_metadata_where("dc", source_institution, clinical_department, time_range, params, document_kind)
    params.append(topk)
    sql = f"""
        SELECT dc.doc_id, dc.title, dc.card_text, dc.publication_date,
               dc.source_institution, dc.clinical_department, d.cleaning_quality, d.cleaning_flags,
               d.source_pdf_text_quality, d.source_pdf_needs_ocr, d.source_pdf_is_scanned,
               d.pdf_text_quality, d.pdf_needs_ocr, d.pdf_is_scanned,
               d.ocr_engine, d.ocr_applied, d.ocr_status, d.ocr_error,
               bm25({table}, {weights}) AS bm25_score
        FROM {table}
        JOIN document_cards dc ON dc.rowid = {table}.rowid
        JOIN documents d ON d.doc_id = dc.doc_id
        WHERE {table} MATCH ? {metadata_sql}
        ORDER BY bm25_score ASC
        LIMIT ?
    """
    with closing(connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "card_text": row["card_text"],
            "fields": {},
            "publication_date": row["publication_date"],
            "source_institution": row["source_institution"],
            "clinical_department": row["clinical_department"],
            "cleaning_quality": row["cleaning_quality"],
            "cleaning_flags": row["cleaning_flags"],
            "source_pdf_text_quality": row["source_pdf_text_quality"],
            "source_pdf_needs_ocr": bool(row["source_pdf_needs_ocr"]),
            "source_pdf_is_scanned": bool(row["source_pdf_is_scanned"]),
            "pdf_text_quality": row["pdf_text_quality"],
            "pdf_needs_ocr": bool(row["pdf_needs_ocr"]),
            "pdf_is_scanned": bool(row["pdf_is_scanned"]),
            "ocr_engine": row["ocr_engine"],
            "ocr_applied": bool(row["ocr_applied"]),
            "ocr_status": row["ocr_status"],
            "ocr_error": row["ocr_error"],
            "score": float(-row["bm25_score"]),
        }
        for row in rows
    ]

def helper_search_document_views_fts(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 50,
    zh: bool = False,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # 视图命中按 doc_id 聚合前保留视图类型和原始名次，供后续融合解释使用。
    table = "document_views_zh_fts" if zh else "document_views_fts"
    weights = "1.5, 4.0, 3.0" if zh else "1.5, 4.0, 1.0, 3.0"
    fts = helper_zh_fts_query(query) if zh else helper_fts_query(query)
    params: list[Any] = [fts]
    metadata_sql = helper_metadata_where("v", source_institution, clinical_department, time_range, params, document_kind)
    params.append(topk)
    sql = f"""
        SELECT v.view_id, v.doc_id, v.view_type, v.priority, v.title, v.text, v.publication_date,
               v.source_institution, v.clinical_department, d.cleaning_quality, d.cleaning_flags,
               d.source_pdf_text_quality, d.source_pdf_needs_ocr, d.source_pdf_is_scanned,
               d.pdf_text_quality, d.pdf_needs_ocr, d.pdf_is_scanned,
               d.ocr_engine, d.ocr_applied, d.ocr_status, d.ocr_error,
               bm25({table}, {weights}) AS bm25_score
        FROM {table}
        JOIN document_views v ON v.rowid = {table}.rowid
        JOIN documents d ON d.doc_id = v.doc_id
        WHERE {table} MATCH ? {metadata_sql}
        ORDER BY bm25_score ASC
        LIMIT ?
    """
    with closing(connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "view_id": row["view_id"],
            "doc_id": row["doc_id"],
            "view_type": row["view_type"],
            "priority": float(row["priority"] or 1.0),
            "title": row["title"],
            "text": row["text"],
            "publication_date": row["publication_date"],
            "source_institution": row["source_institution"],
            "clinical_department": row["clinical_department"],
            "cleaning_quality": row["cleaning_quality"],
            "cleaning_flags": row["cleaning_flags"],
            "source_pdf_text_quality": row["source_pdf_text_quality"],
            "source_pdf_needs_ocr": bool(row["source_pdf_needs_ocr"]),
            "source_pdf_is_scanned": bool(row["source_pdf_is_scanned"]),
            "pdf_text_quality": row["pdf_text_quality"],
            "pdf_needs_ocr": bool(row["pdf_needs_ocr"]),
            "pdf_is_scanned": bool(row["pdf_is_scanned"]),
            "ocr_engine": row["ocr_engine"],
            "ocr_applied": bool(row["ocr_applied"]),
            "ocr_status": row["ocr_status"],
            "ocr_error": row["ocr_error"],
            "score": float(-row["bm25_score"]),
        }
        for row in rows
    ]

def helper_fuse_document_results(result_lists: list[list[dict[str, Any]]], topk: int) -> list[dict[str, Any]]:
    rank_lists = [[item["doc_id"] for item in results] for results in result_lists if results]
    if not rank_lists:
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for item in results:
            by_id.setdefault(item["doc_id"], item)
    fused = rrf_fusion(rank_lists)
    output = []
    for doc_id, score in fused[:topk]:
        item = dict(by_id[doc_id])
        item["score"] = score
        output.append(item)
    return output


def helper_view_results_as_documents(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for item in results:
        doc_id = item.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        output.append(
            {
                "doc_id": doc_id,
                "title": item.get("title", ""),
                "abstract": "",
                "publication_date": item.get("publication_date", "unknown"),
                "source_institution": item.get("source_institution", "Unknown"),
                "clinical_department": item.get("clinical_department", "???"),
                "cleaning_quality": item.get("cleaning_quality", ""),
                "cleaning_flags": item.get("cleaning_flags", ""),
                "source_pdf_text_quality": item.get("source_pdf_text_quality", ""),
                "source_pdf_needs_ocr": bool(item.get("source_pdf_needs_ocr")),
                "source_pdf_is_scanned": bool(item.get("source_pdf_is_scanned")),
                "pdf_text_quality": item.get("pdf_text_quality", ""),
                "pdf_needs_ocr": bool(item.get("pdf_needs_ocr")),
                "pdf_is_scanned": bool(item.get("pdf_is_scanned")),
                "ocr_engine": item.get("ocr_engine", ""),
                "ocr_applied": bool(item.get("ocr_applied")),
                "ocr_status": item.get("ocr_status", ""),
                "ocr_error": item.get("ocr_error", ""),
                "matched_view_id": item.get("view_id"),
                "matched_view_type": item.get("view_type"),
                "score": float(item.get("score") or 0.0),
            }
        )
    return output


def search_documents_sqlite(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 10,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    pool_size = max(50, topk)
    card_lexical = helper_search_document_cards_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=False, document_kind=document_kind)
    card_zh = helper_search_document_cards_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=True, document_kind=document_kind)
    view_lexical = helper_view_results_as_documents(
        helper_search_document_views_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=False, document_kind=document_kind)
    )
    view_zh = helper_view_results_as_documents(
        helper_search_document_views_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=True, document_kind=document_kind)
    )
    # 中文分词表与通用词法表分别召回，再按文档 ID 融合，兼顾中英文和混合查询。
    return helper_fuse_document_results([card_zh, card_lexical, view_zh, view_lexical], topk)


def search_document_cards_sqlite(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 50,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    pool_size = max(50, topk)
    lexical = helper_search_document_cards_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=False, document_kind=document_kind)
    zh_results = helper_search_document_cards_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=True, document_kind=document_kind)
    return helper_fuse_document_results([zh_results, lexical], topk)


def helper_fuse_view_results(result_lists: list[list[dict[str, Any]]], topk: int) -> list[dict[str, Any]]:
    rank_lists = [[item["view_id"] for item in results] for results in result_lists if results]
    if not rank_lists:
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for item in results:
            by_id.setdefault(item["view_id"], item)
    fused = rrf_fusion(rank_lists)
    output = []
    for view_id, score in fused[:topk]:
        item = dict(by_id[view_id])
        item["score"] = score
        output.append(item)
    return output


def search_document_views_sqlite(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 80,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    pool_size = max(80, topk)
    lexical = helper_search_document_views_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=False, document_kind=document_kind)
    zh_results = helper_search_document_views_fts(query, db_path, source_institution, clinical_department, time_range, pool_size, zh=True, document_kind=document_kind)
    return helper_fuse_view_results([zh_results, lexical], topk)


def helper_retrieve_chunks_fts(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 50,
    exclude_reference_sections: bool = True,
    zh: bool = False,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    # FTS 只召回分块候选，参考文献与文档类型过滤必须在返回前完成。
    table = "chunks_zh_fts" if zh else "chunks_fts"
    weights = "4.0, 2.0, 1.0" if zh else "4.0, 2.0, 2.0, 1.0"
    fts = helper_zh_fts_query(query) if zh else helper_fts_query(query)
    params: list[Any] = [fts]
    clauses = []
    if exclude_reference_sections:
        clauses.append("c.is_reference_section = 0")
    metadata_sql = helper_metadata_where("c", source_institution, clinical_department, time_range, params, document_kind)
    base_sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    params.append(topk)
    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.content, c.title, c.publication_date, c.source_institution, c.clinical_department,
               c.section_path, c.chunk_index, c.retrieval_key, d.source_file, d.markdown_clean_path,
               c.retrieval_text, c.chunk_type, c.token_count, c.is_background, c.is_reference_section,
               bm25({table}, {weights}) AS bm25_score
        FROM {table}
        JOIN chunks c ON c.rowid = {table}.rowid
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE {table} MATCH ? {base_sql} {metadata_sql}
        ORDER BY bm25_score ASC
        LIMIT ?
    """
    with closing(connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        try:
            section_path = json.loads(row["section_path"] or "[]")
        except json.JSONDecodeError:
            section_path = []
        item = {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "content": row["content"],
            "title": row["title"],
            "publication_date": row["publication_date"],
            "source_institution": row["source_institution"],
            "clinical_department": row["clinical_department"],
            "section_path": section_path,
            "chunk_index": row["chunk_index"],
            "score": float(-row["bm25_score"]),
            "retrieval_key": row["retrieval_key"],
            "source_file": row["source_file"],
            "markdown_clean_path": row["markdown_clean_path"],
            "retrieval_text": row["retrieval_text"],
            "chunk_type": row["chunk_type"],
            "token_count": row["token_count"],
            "is_background": bool(row["is_background"]),
            "is_reference_section": bool(row["is_reference_section"]),
        }
        item.update(helper_department_metadata(row["clinical_department"]))
        results.append(helper_enrich_chunk_metadata(item))
    return results

def helper_fuse_chunk_results(result_lists: list[list[dict[str, Any]]], topk: int) -> list[dict[str, Any]]:
    rank_lists = [[item["chunk_id"] for item in results] for results in result_lists if results]
    if not rank_lists:
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for item in results:
            by_id.setdefault(item["chunk_id"], item)
    fused = rrf_fusion(rank_lists)
    output = []
    for chunk_id, score in fused[:topk]:
        item = dict(by_id[chunk_id])
        item["score"] = score
        output.append(item)
    return output


def retrieve_chunks_sqlite(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_institution: str | None = None,
    clinical_department: str | None = None,
    time_range: str | dict[str, str] | None = None,
    topk: int = 5,
    exclude_reference_sections: bool = True,
    document_kind: str | None = None,
) -> list[dict[str, Any]]:
    pool_size = max(50, topk)
    lexical = helper_retrieve_chunks_fts(
        query, db_path, source_institution, clinical_department, time_range, pool_size, exclude_reference_sections, zh=False, document_kind=document_kind
    )
    zh_results = helper_retrieve_chunks_fts(
        query, db_path, source_institution, clinical_department, time_range, pool_size, exclude_reference_sections, zh=True, document_kind=document_kind
    )
    return helper_fuse_chunk_results([zh_results, lexical], topk)


def read_document_sqlite(
    doc_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    title: str | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    # doc_id 与标题是互斥定位方式，读取路径必须经过存在性检查后才返回正文。
    if not doc_id and not title:
        raise ValueError("read_document_sqlite requires either doc_id or title")
    if doc_id:
        sql = """
            SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                   source_file, content_md, markdown_clean_path
            FROM documents
            WHERE doc_id=?
        """
        params: tuple[Any, ...] = (doc_id,)
    else:
        query_title = (title or "").strip()
        sql = """
            SELECT doc_id, title, publication_date, source_institution, clinical_department, document_kind,
                   source_file, content_md, markdown_clean_path
            FROM documents
            WHERE title = ? OR title LIKE ?
            ORDER BY CASE WHEN title = ? THEN 0 ELSE 1 END, publication_date DESC, length(title) ASC
            LIMIT 1
        """
        params = (query_title, f"%{query_title}%", query_title)
    with closing(connect(db_path)) as conn:
        row = conn.execute(sql, params).fetchone()
    if not row:
        raise KeyError(f"Document not found: {doc_id or title}")
    content = row["content_md"]
    clean_path = Path(row["markdown_clean_path"] or "")
    if not content:
        candidate_paths = [clean_path]
        if not clean_path.exists():
            candidate_paths.append(Path(db_path).parent.parent / "markdown_clean" / f"{row['doc_id']}.md")
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                content = candidate_path.read_text(encoding="utf-8", errors="replace")
                break
    if max_chars and max_chars > 0:
        content = content[:max_chars]
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "publication_date": row["publication_date"],
        "source_institution": row["source_institution"],
        "clinical_department": row["clinical_department"],
        "document_kind": row["document_kind"],
        "source_file": row["source_file"],
        "markdown_clean_path": row["markdown_clean_path"],
        "content": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "init", "stats", "search", "retrieve"])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--query", default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--source-institution")
    parser.add_argument("--clinical-department")
    parser.add_argument("--time-range")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_sqlite_store(args.data_dir, args.db_path, args.batch_size, reset=True)
    elif args.command == "init":
        payload = init_schema(args.db_path, reset=False)
    elif args.command == "stats":
        payload = stats(args.db_path)
    elif args.command == "search":
        payload = search_documents_sqlite(
            args.query,
            args.db_path,
            source_institution=args.source_institution,
            clinical_department=args.clinical_department,
            time_range=args.time_range,
            topk=args.topk,
        )
    else:
        payload = retrieve_chunks_sqlite(
            args.query,
            args.db_path,
            source_institution=args.source_institution,
            clinical_department=args.clinical_department,
            time_range=args.time_range,
            topk=args.topk,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
