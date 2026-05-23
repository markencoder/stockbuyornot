from pathlib import Path

import pytest

from stockbuyornot.auth import (
    authenticate_user,
    create_user,
    hash_password,
    sanitize_user_key,
    subscription_is_active,
    update_subscription_status,
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
    assert signed_in.subscription_status == "free"
    assert not subscription_is_active(signed_in)


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
    assert signed_in.subscription_expires_at == "2026-12-31"


def test_sanitize_user_key_keeps_user_storage_path_safe():
    assert sanitize_user_key("Trader+VIP@Example.COM") == "trader_vip_example.com"
