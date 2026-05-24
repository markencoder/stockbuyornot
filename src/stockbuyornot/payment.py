from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PAYMENT_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "payment"


@dataclass(frozen=True)
class PaymentConfig:
    provider: str
    amount_cny: str
    qr_image_url: str
    alipay_qr_image: str
    wechatpay_qr_image: str
    support_contact: str
    billing_enforced: bool


def payment_config_from_env() -> PaymentConfig:
    return PaymentConfig(
        provider=os.environ.get("STOCKBUYORNOT_PAYMENT_PROVIDER", "manual_qr"),
        amount_cny=os.environ.get("STOCKBUYORNOT_PAYMENT_AMOUNT_CNY", "99"),
        qr_image_url=os.environ.get("STOCKBUYORNOT_PAYMENT_QR_URL", ""),
        alipay_qr_image=os.environ.get("STOCKBUYORNOT_ALIPAY_QR_URL", "") or str(DEFAULT_PAYMENT_ASSET_DIR / "alipay.png"),
        wechatpay_qr_image=os.environ.get("STOCKBUYORNOT_WECHATPAY_QR_URL", "") or str(DEFAULT_PAYMENT_ASSET_DIR / "wechatpay.png"),
        support_contact=os.environ.get("STOCKBUYORNOT_SUPPORT_CONTACT", ""),
        billing_enforced=os.environ.get("STOCKBUYORNOT_BILLING_ENFORCED", "").lower() in {"1", "true", "yes"},
    )


def payment_qr_images(config: PaymentConfig) -> list[tuple[str, str]]:
    images = [
        ("支付宝", config.alipay_qr_image),
        ("微信", config.wechatpay_qr_image),
    ]
    if config.qr_image_url:
        images.append((config.provider, config.qr_image_url))
    return [(label, source) for label, source in images if _image_source_available(source)]


def payment_reference(user_id: int, email: str) -> str:
    return f"SBO-{user_id}-{email.split('@', 1)[0]}"


def _image_source_available(source: str) -> bool:
    if not source:
        return False
    if source.startswith(("http://", "https://")):
        return True
    return Path(source).exists()
