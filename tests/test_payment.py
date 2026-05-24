from pathlib import Path

from stockbuyornot.payment import payment_config_from_env, payment_qr_images


def test_default_payment_qr_images_are_bundled(monkeypatch):
    monkeypatch.delenv("STOCKBUYORNOT_PAYMENT_QR_URL", raising=False)
    monkeypatch.delenv("STOCKBUYORNOT_ALIPAY_QR_URL", raising=False)
    monkeypatch.delenv("STOCKBUYORNOT_WECHATPAY_QR_URL", raising=False)

    images = payment_qr_images(payment_config_from_env())

    labels = [label for label, _ in images]
    sources = [source for _, source in images]
    assert labels == ["支付宝", "微信"]
    assert all(Path(source).exists() for source in sources)


def test_external_payment_qr_urls_override_bundled_assets(monkeypatch):
    monkeypatch.setenv("STOCKBUYORNOT_ALIPAY_QR_URL", "https://example.com/alipay.png")
    monkeypatch.setenv("STOCKBUYORNOT_WECHATPAY_QR_URL", "https://example.com/wechatpay.png")

    images = payment_qr_images(payment_config_from_env())

    assert images[:2] == [
        ("支付宝", "https://example.com/alipay.png"),
        ("微信", "https://example.com/wechatpay.png"),
    ]
