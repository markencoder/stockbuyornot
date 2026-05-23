import pandas as pd

from stockbuyornot.data.providers import AkshareProvider


class FakeResponse:
    def __init__(self, market_symbol):
        self.market_symbol = market_symbol

    def json(self):
        return {
            "data": {
                self.market_symbol: {
                    "day": [
                        ["2026-05-20", "4100", "4120", "4130", "4090", "1000"],
                        ["2026-05-21", "4120", "4174", "4200", "4074", "1200"],
                    ]
                }
            }
        }


class FakeProvider(AkshareProvider):
    def __init__(self):
        super().__init__()
        self.requested_params = []

    def _request_tencent(self, url, params):
        self.requested_params.append(params["param"])
        market_symbol = params["param"].split(",", 1)[0]
        return FakeResponse(market_symbol)


def test_tencent_index_daily_uses_real_shanghai_index_symbol():
    provider = FakeProvider()

    result = provider._tencent_index_daily("000001", "20260501", "20260521")

    assert provider.requested_params[0].startswith("sh000001,day,")
    assert len(result) == 2
    assert result["symbol"].iloc[-1] == "000001"
    assert result["close"].iloc[-1] == 4174.0


def test_tencent_index_daily_uses_real_shenzhen_index_symbol():
    provider = FakeProvider()

    result = provider._tencent_index_daily("399001", "20260501", "20260521")

    assert provider.requested_params[0].startswith("sz399001,day,")
    assert result["symbol"].iloc[-1] == "399001"
