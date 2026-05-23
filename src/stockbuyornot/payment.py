from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PaymentConfig:
    provider: str
    amount_cny: str
    qr_image_url: str
    support_contact: str
    billing_enforced: bool


def payment_config_from_env() -> PaymentConfig:
    return PaymentConfig(
        provider=os.environ.get("STOCKBUYORNOT_PAYMENT_PROVIDER", "manual_qr"),
        amount_cny=os.environ.get("STOCKBUYORNOT_PAYMENT_AMOUNT_CNY", "99"),
        qr_image_url=os.environ.get("STOCKBUYORNOT_PAYMENT_QR_URL", ""),
        support_contact=os.environ.get("STOCKBUYORNOT_SUPPORT_CONTACT", ""),
        billing_enforced=os.environ.get("STOCKBUYORNOT_BILLING_ENFORCED", "").lower() in {"1", "true", "yes"},
    )


def payment_reference(user_id: int, email: str) -> str:
    return f"SBO-{user_id}-{email.split('@', 1)[0]}"
