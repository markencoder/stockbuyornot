from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "app.db"
PASSWORD_ITERATIONS = 260_000
DEFAULT_SESSION_HOURS = 12.0
MEMBER_STATUSES = {"active", "member"}
ADMIN_STATUS = "admin"
TRIAL_STATUS = "trial"
DISABLED_STATUS = "disabled"
TRIAL_DAILY_LIMITS = {
    "single_diagnosis": 10,
    "stock_pool_scan": 3,
    "backtest": 3,
    "follow_trade": 3,
}
TRIAL_STORAGE_LIMITS = {
    "watchlist": 10,
    "purchased": 3,
}


@dataclass(frozen=True)
class User:
    id: int
    email: str
    display_name: str
    subscription_status: str
    subscription_expires_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


def data_root() -> Path:
    return Path(os.environ.get("STOCKBUYORNOT_DATA_DIR", DEFAULT_DATA_DIR))


def database_path() -> Path:
    if "STOCKBUYORNOT_DB_PATH" in os.environ:
        return Path(os.environ["STOCKBUYORNOT_DB_PATH"])
    return data_root() / "app.db"


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
                subscription_status TEXT NOT NULL DEFAULT 'trial',
                subscription_expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_usage (
                user_id INTEGER NOT NULL,
                usage_date TEXT NOT NULL,
                feature TEXT NOT NULL,
                used_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, usage_date, feature),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_subscription_status
            ON users(subscription_status)
            """
        )
        connection.execute(
            """
            UPDATE users
            SET subscription_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE subscription_status IN ('free', 'expired')
            """,
            (TRIAL_STATUS,),
        )


def session_hours_from_env() -> float:
    raw = os.environ.get("STOCKBUYORNOT_SESSION_HOURS", str(DEFAULT_SESSION_HOURS))
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SESSION_HOURS
    return min(max(hours, 0.25), 720.0)


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


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


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
    downgrade_expired_memberships(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    if row is None or not verify_password(password, str(row["password_hash"])):
        return None
    user = _row_to_user(row)
    if is_disabled_user(user):
        return None
    return user


def get_user_by_id(user_id: int, db_path: Path | None = None) -> User | None:
    initialize_auth_db(db_path)
    downgrade_expired_memberships(db_path)
    with connect(db_path) as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row is not None else None


def get_user_by_email(email: str, db_path: Path | None = None) -> User | None:
    initialize_auth_db(db_path)
    downgrade_expired_memberships(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()
    return _row_to_user(row) if row is not None else None


def list_users(limit: int = 100, db_path: Path | None = None) -> list[User]:
    initialize_auth_db(db_path)
    downgrade_expired_memberships(db_path)
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


def create_auth_session(
    user_id: int,
    db_path: Path | None = None,
    *,
    now: datetime | None = None,
    duration_hours: float | None = None,
) -> tuple[str, str]:
    initialize_auth_db(db_path)
    issued_at = now or datetime.now(timezone.utc)
    hours = session_hours_from_env() if duration_hours is None else duration_hours
    expires_at = issued_at + timedelta(hours=float(hours))
    token = secrets.token_urlsafe(32)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO auth_sessions(token_hash, user_id, expires_at)
            VALUES (?, ?, ?)
            """,
            (hash_session_token(token), int(user_id), expires_at.isoformat()),
        )
    return token, expires_at.isoformat()


def get_user_by_session_token(
    token: str | None,
    db_path: Path | None = None,
    *,
    now: datetime | None = None,
) -> User | None:
    if not token:
        return None
    initialize_auth_db(db_path)
    downgrade_expired_memberships(db_path)
    checked_at = now or datetime.now(timezone.utc)
    token_hash = hash_session_token(str(token))
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?
            """,
            (token_hash, checked_at.isoformat()),
        ).fetchone()
        if row is None:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
            return None
    user = _row_to_user(row)
    if is_disabled_user(user):
        revoke_auth_session(token, db_path)
        return None
    return user


def revoke_auth_session(token: str | None, db_path: Path | None = None) -> None:
    if not token:
        return
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (hash_session_token(str(token)),))


def downgrade_expired_memberships(db_path: Path | None = None, today: date | None = None) -> None:
    today_text = (today or date.today()).isoformat()
    initialize_auth_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE users
            SET subscription_status = ?, subscription_expires_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE subscription_status IN ('active', 'member')
              AND subscription_expires_at IS NOT NULL
              AND subscription_expires_at < ?
            """,
            (TRIAL_STATUS, today_text),
        )


def admin_emails_from_env() -> set[str]:
    raw = os.environ.get("STOCKBUYORNOT_ADMIN_EMAILS", "")
    return {normalize_email(item) for item in re.split(r"[,;\s]+", raw) if item.strip()}


def is_admin_user(user: User) -> bool:
    if is_disabled_user(user):
        return False
    return user.subscription_status == ADMIN_STATUS or normalize_email(user.email) in admin_emails_from_env()


def is_disabled_user(user: User) -> bool:
    return user.subscription_status == DISABLED_STATUS


def is_member_user(user: User) -> bool:
    if is_disabled_user(user):
        return False
    return is_admin_user(user) or user.subscription_status in MEMBER_STATUSES


def subscription_is_active(user: User) -> bool:
    if is_disabled_user(user):
        return False
    return is_admin_user(user) or user.subscription_status in MEMBER_STATUSES | {TRIAL_STATUS}


def user_tier(user: User) -> str:
    if is_disabled_user(user):
        return "disabled"
    if is_admin_user(user):
        return "admin"
    if user.subscription_status in MEMBER_STATUSES:
        return "member"
    return "trial"


def trial_daily_limit(feature: str) -> int | None:
    return TRIAL_DAILY_LIMITS.get(feature)


def trial_storage_limit(kind: str) -> int | None:
    return TRIAL_STORAGE_LIMITS.get(kind)


def usage_count(
    user_id: int,
    feature: str,
    usage_date: date | None = None,
    db_path: Path | None = None,
) -> int:
    initialize_auth_db(db_path)
    day = (usage_date or date.today()).isoformat()
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT used_count
            FROM user_usage
            WHERE user_id = ? AND usage_date = ? AND feature = ?
            """,
            (user_id, day, feature),
        ).fetchone()
    return 0 if row is None else int(row["used_count"])


def consume_trial_usage(
    user: User,
    feature: str,
    amount: int = 1,
    usage_date: date | None = None,
    db_path: Path | None = None,
) -> tuple[bool, int, int | None]:
    limit = trial_daily_limit(feature)
    if is_disabled_user(user):
        return False, 0, limit
    if user_tier(user) != "trial" or limit is None:
        return True, 0, limit

    initialize_auth_db(db_path)
    day = (usage_date or date.today()).isoformat()
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT used_count
            FROM user_usage
            WHERE user_id = ? AND usage_date = ? AND feature = ?
            """,
            (user.id, day, feature),
        ).fetchone()
        current = 0 if row is None else int(row["used_count"])
        if current + amount > limit:
            return False, current, limit
        connection.execute(
            """
            INSERT INTO user_usage(user_id, usage_date, feature, used_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, usage_date, feature) DO UPDATE SET
                used_count = user_usage.used_count + excluded.used_count,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user.id, day, feature, int(amount)),
        )
    return True, current + amount, limit


def trial_usage_snapshot(user: User, db_path: Path | None = None) -> dict[str, tuple[int, int]]:
    if user_tier(user) != "trial":
        return {}
    return {feature: (usage_count(user.id, feature, db_path=db_path), limit) for feature, limit in TRIAL_DAILY_LIMITS.items()}



def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        subscription_status=str(row["subscription_status"]),
        subscription_expires_at=row["subscription_expires_at"],
        created_at=row["created_at"] if "created_at" in row.keys() else None,
        updated_at=row["updated_at"] if "updated_at" in row.keys() else None,
    )


def _valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))
