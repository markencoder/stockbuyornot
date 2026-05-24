from pathlib import Path

import pytest

from stockbuyornot.auth import (
    authenticate_user,
    consume_trial_usage,
    create_user,
    data_root,
    database_path,
    downgrade_expired_memberships,
    hash_password,
    is_member_user,
    sanitize_user_key,
    subscription_is_active,
    update_subscription_status,
    usage_count,
    user_tier,
    verify_password,
)


def test_password_hash_roundtrip_does_not_store_plaintext():
    stored = hash_password("strong-pass-123")

    assert "strong-pass-123" not in stored
    assert verify_password("strong-pass-123", stored)
    assert not verify_password("wrong-pass-123", stored)


def test_create_and_authenticate_user(tmp_path: Path):
    db_path = tmp_path / "app.db"

    created = create_user("Trader@example.com", "strong-pass-123", "Trader", db_path)
    signed_in = authenticate_user("trader@example.com", "strong-pass-123", db_path)

    assert signed_in is not None
    assert signed_in.id == created.id
    assert signed_in.display_name == "Trader"
    assert signed_in.subscription_status == "trial"
    assert subscription_is_active(signed_in)
    assert user_tier(signed_in) == "trial"


def test_duplicate_email_is_rejected(tmp_path: Path):
    db_path = tmp_path / "app.db"
    create_user("trader@example.com", "strong-pass-123", db_path=db_path)

    with pytest.raises(ValueError, match="已经注册"):
        create_user("TRADER@example.com", "strong-pass-123", db_path=db_path)


def test_subscription_status_can_be_upgraded(tmp_path: Path):
    db_path = tmp_path / "app.db"
    user = create_user("paid@example.com", "strong-pass-123", db_path=db_path)

    update_subscription_status(user.id, "active", "2026-12-31", db_path)
    signed_in = authenticate_user("paid@example.com", "strong-pass-123", db_path)

    assert signed_in is not None
    assert subscription_is_active(signed_in)
    assert is_member_user(signed_in)
    assert signed_in.subscription_expires_at == "2026-12-31"


def test_expired_member_is_downgraded_to_trial(tmp_path: Path):
    db_path = tmp_path / "app.db"
    user = create_user("expired@example.com", "strong-pass-123", db_path=db_path)

    update_subscription_status(user.id, "active", "2026-01-01", db_path)
    downgrade_expired_memberships(db_path)
    signed_in = authenticate_user("expired@example.com", "strong-pass-123", db_path)

    assert signed_in is not None
    assert signed_in.subscription_status == "trial"
    assert signed_in.subscription_expires_at is None


def test_trial_usage_is_limited_per_day(tmp_path: Path):
    db_path = tmp_path / "app.db"
    user = create_user("trial@example.com", "strong-pass-123", db_path=db_path)

    for index in range(10):
        allowed, used, limit = consume_trial_usage(user, "single_diagnosis", db_path=db_path)
        assert allowed
        assert used == index + 1
        assert limit == 10

    allowed, used, limit = consume_trial_usage(user, "single_diagnosis", db_path=db_path)

    assert not allowed
    assert used == 10
    assert limit == 10
    assert usage_count(user.id, "single_diagnosis", db_path=db_path) == 10


def test_member_usage_is_unlimited(tmp_path: Path):
    db_path = tmp_path / "app.db"
    user = create_user("member@example.com", "strong-pass-123", db_path=db_path)
    update_subscription_status(user.id, "active", "2026-12-31", db_path)
    member = authenticate_user("member@example.com", "strong-pass-123", db_path)

    assert member is not None
    for _ in range(12):
        allowed, used, limit = consume_trial_usage(member, "single_diagnosis", db_path=db_path)
        assert allowed
        assert used == 0
        assert limit == 10


def test_sanitize_user_key_keeps_user_storage_path_safe():
    assert sanitize_user_key("Trader+VIP@Example.COM") == "trader_vip_example.com"


def test_data_root_env_controls_default_database_path(tmp_path: Path, monkeypatch):
    persistent_dir = tmp_path / "persistent-data"

    monkeypatch.setenv("STOCKBUYORNOT_DATA_DIR", str(persistent_dir))
    monkeypatch.delenv("STOCKBUYORNOT_DB_PATH", raising=False)

    assert data_root() == persistent_dir
    assert database_path() == persistent_dir / "app.db"


def test_db_path_env_overrides_data_root(tmp_path: Path, monkeypatch):
    persistent_dir = tmp_path / "persistent-data"
    explicit_db = tmp_path / "db" / "custom.db"

    monkeypatch.setenv("STOCKBUYORNOT_DATA_DIR", str(persistent_dir))
    monkeypatch.setenv("STOCKBUYORNOT_DB_PATH", str(explicit_db))

    assert data_root() == persistent_dir
    assert database_path() == explicit_db
