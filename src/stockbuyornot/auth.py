from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("data/app.db")
PASSWORD_ITERATIONS = 260_000


@dataclass(frozen=True)
class User:
    id: int
    email: str
    display_name: str
    subscription_status: str
    subscription_expires_at: str | None = None


def database_path() -> Path:
    return Path(os.environ.get("STOCKBUYORNOT_DB_PATH", DEFAULT_DB_PATH))


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_db(db_path: Path | None = None) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                subscription_status TEXT NOT NULL DEFAULT 'free',
                subscription_expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_subscription_status
            ON users(subscription_status)
            """
        )


def normalize_email(email: str) -> str:
    return email.strip().lower()


def sanitize_user_key(value: str) -> str:
    normalized = normalize_email(value)
    safe = re.sub(r"[^a-z0-9_.-]+", "_", normalized)
    return safe.strip("._-") or "user"


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations_text),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_user(email: str, password: str, display_name: str = "", db_path: Path | None = None) -> User:
    normalized = normalize_email(email)
    name = display_name.strip() or normalized.split("@", 1)[0]
    if not _valid_email(normalized):
        raise ValueError("请输入有效邮箱。")
    if len(password) < 8:
        raise ValueError("密码至少需要 8 位。")

    initialize_auth_db(db_path)
    try:
        with connect(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users(email, display_name, password_hash)
                VALUES (?, ?, ?)
                """,
                (normalized, name, hash_password(password)),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("这个邮箱已经注册过。") from exc
    user = get_user_by_id(user_id, db_path)
    if user is None:
        raise RuntimeError("用户创建失败。")
    return user


def authenticate_user(email: str, password: str, db_path: Path | None = None) -> User | None:
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    if row is None or not verify_password(password, str(row["password_hash"])):
        return None
    return _row_to_user(row)


def get_user_by_id(user_id: int, db_path: Path | None = None) -> User | None:
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row is not None else None


def get_user_by_email(email: str, db_path: Path | None = None) -> User | None:
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    return _row_to_user(row) if row is not None else None


def list_users(limit: int = 100, db_path: Path | None = None) -> list[User]:
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM users
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_row_to_user(row) for row in rows]


def update_subscription_status(
    user_id: int,
    status: str,
    expires_at: str | None = None,
    db_path: Path | None = None,
) -> None:
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE users
            SET subscription_status = ?, subscription_expires_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, expires_at, user_id),
        )


def admin_emails_from_env() -> set[str]:
    raw = os.environ.get("STOCKBUYORNOT_ADMIN_EMAILS", "")
    return {normalize_email(item) for item in re.split(r"[,;\s]+", raw) if item.strip()}


def is_admin_user(user: User) -> bool:
    return user.subscription_status == "admin" or normalize_email(user.email) in admin_emails_from_env()


def subscription_is_active(user: User) -> bool:
    return is_admin_user(user) or user.subscription_status in {"active", "trial"}


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        subscription_status=str(row["subscription_status"]),
        subscription_expires_at=row["subscription_expires_at"],
    )


def _valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))
