"""SQLite persistence with WAL, transactions, and one-time JSON migration."""

import json
import sqlite3
from pathlib import Path


class SQLiteStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.path = data_dir / "knowledgepilot.db"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.data_dir.mkdir(exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL, stored_filename TEXT, status TEXT NOT NULL DEFAULT 'ready',
                    progress INTEGER NOT NULL DEFAULT 100, error_detail TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    document TEXT NOT NULL, content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_task ON documents(task_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_task ON chunks(task_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, task_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if "progress" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN progress INTEGER NOT NULL DEFAULT 100")
            if "error_detail" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN error_detail TEXT")
            message_columns = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
            if "task_id" not in message_columns:
                db.execute("ALTER TABLE messages ADD COLUMN task_id TEXT")
        self._migrate_json_once()

    def _json(self, name: str, default):
        path = self.data_dir / name
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default

    def _migrate_json_once(self) -> None:
        with self._connect() as db:
            if db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]:
                return
            tasks = self._json("tasks.json", [])
            documents = self._json("documents.json", [])
            chunks = self._json("index.json", [])
            sessions = self._json("sessions.json", {})
            with db:
                db.executemany("INSERT OR IGNORE INTO tasks(id, name) VALUES (?, ?)", [(item["id"], item["name"]) for item in tasks])
                db.executemany(
                    "INSERT OR IGNORE INTO documents(id, task_id, filename, stored_filename) VALUES (?, ?, ?, ?)",
                    [(item["id"], item["task_id"], item["filename"], item.get("stored_filename")) for item in documents],
                )
                db.executemany(
                    "INSERT OR IGNORE INTO chunks(id, task_id, document_id, document, content) VALUES (?, ?, ?, ?, ?)",
                    [(item["id"], item.get("task_id", ""), item.get("document_id", ""), item["document"], item["content"]) for item in chunks if item.get("task_id") and item.get("document_id")],
                )
                for session_id, messages in sessions.items():
                    db.executemany("INSERT INTO messages(session_id, role, content, task_id) VALUES (?, ?, ?, ?)", [(session_id, message["role"], message["content"], tasks[0]["id"] if len(tasks) == 1 else None) for message in messages])

    def task(self, task_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT id, name FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def tasks(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT t.id, t.name, COUNT(DISTINCT d.id) document_count, COUNT(c.id) chunk_count
                FROM tasks t LEFT JOIN documents d ON d.task_id=t.id LEFT JOIN chunks c ON c.task_id=t.id
                GROUP BY t.id ORDER BY t.rowid""").fetchall()
        return [dict(row) for row in rows]

    def add_task(self, task_id: str, name: str) -> None:
        with self._connect() as db, db:
            db.execute("INSERT INTO tasks(id, name) VALUES (?, ?)", (task_id, name))

    def documents(self, task_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("""SELECT d.id, d.filename, d.status, d.progress, d.error_detail, COUNT(c.id) chunks FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id WHERE d.task_id=? GROUP BY d.id ORDER BY d.rowid""", (task_id,)).fetchall()
        return [dict(row) for row in rows]

    def document(self, task_id: str, document_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM documents WHERE task_id=? AND id=?", (task_id, document_id)).fetchone()
        return dict(row) if row else None

    def add_document_with_chunks(self, document: dict, chunks: list[dict]) -> None:
        with self._connect() as db, db:
            db.execute("INSERT INTO documents(id, task_id, filename, stored_filename, status) VALUES (?, ?, ?, ?, 'ready')", (document["id"], document["task_id"], document["filename"], document["stored_filename"]))
            db.executemany("INSERT INTO chunks(id, task_id, document_id, document, content) VALUES (?, ?, ?, ?, ?)", [(chunk["id"], chunk["task_id"], chunk["document_id"], chunk["document"], chunk["content"]) for chunk in chunks])

    def create_processing_document(self, document: dict) -> None:
        with self._connect() as db, db:
            db.execute("INSERT INTO documents(id, task_id, filename, stored_filename, status, progress) VALUES (?, ?, ?, ?, 'processing', 5)", (document["id"], document["task_id"], document["filename"], document["stored_filename"]))

    def complete_document(self, document_id: str, chunks: list[dict]) -> None:
        with self._connect() as db, db:
            db.executemany("INSERT INTO chunks(id, task_id, document_id, document, content) VALUES (?, ?, ?, ?, ?)", [(chunk["id"], chunk["task_id"], chunk["document_id"], chunk["document"], chunk["content"]) for chunk in chunks])
            db.execute("UPDATE documents SET status='ready', progress=100, error_detail=NULL WHERE id=?", (document_id,))

    def set_document_status(self, document_id: str, status: str, progress: int | None = None, error_detail: str | None = None) -> None:
        with self._connect() as db, db:
            db.execute("UPDATE documents SET status=?, progress=COALESCE(?, progress), error_detail=? WHERE id=?", (status, progress, error_detail, document_id))

    def start_reindex(self, task_id: str, document_id: str) -> dict | None:
        with self._connect() as db, db:
            row = db.execute("SELECT * FROM documents WHERE task_id=? AND id=?", (task_id, document_id)).fetchone()
            if not row:
                return None
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.execute("UPDATE documents SET status='processing', progress=5, error_detail=NULL WHERE id=?", (document_id,))
        return dict(row)

    def delete_task(self, task_id: str) -> list[str] | None:
        with self._connect() as db, db:
            task = db.execute("SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not task:
                return None
            files = [row[0] for row in db.execute("SELECT stored_filename FROM documents WHERE task_id=?", (task_id,)).fetchall() if row[0]]
            db.execute("DELETE FROM messages WHERE task_id=?", (task_id,))
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        return files

    def delete_document_with_chunks(self, task_id: str, document_id: str) -> dict | None:
        with self._connect() as db, db:
            row = db.execute("SELECT * FROM documents WHERE task_id=? AND id=?", (task_id, document_id)).fetchone()
            if not row:
                return None
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return dict(row)

    def chunks(self, task_id: str | None = None) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT id, task_id, document_id, document, content FROM chunks" + (" WHERE task_id=?" if task_id else ""), (task_id,) if task_id else ()).fetchall()
        return [dict(row) for row in rows]

    def messages(self, session_id: str, task_id: str | None = None) -> list[dict]:
        with self._connect() as db:
            query = "SELECT role, content FROM messages WHERE session_id=?"
            parameters: tuple[str, ...] = (session_id,)
            if task_id is not None:
                query += " AND task_id=?"
                parameters = (session_id, task_id)
            rows = db.execute(query + " ORDER BY id", parameters).fetchall()
        return [dict(row) for row in rows]

    def append_message(self, session_id: str, role: str, content: str, task_id: str | None = None) -> None:
        with self._connect() as db, db:
            db.execute("INSERT INTO messages(session_id, role, content, task_id) VALUES (?, ?, ?, ?)", (session_id, role, content, task_id))
