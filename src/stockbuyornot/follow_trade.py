from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class ForeignMove:
    symbol: str
    name: str
    market: str
    pct_change: float


@dataclass(frozen=True)
class UpstreamLink:
    foreign_symbol: str
    foreign_name: str
    market: str
    sector: str
    ashare_symbol: str
    ashare_name: str
    upstream_role: str
    relation_strength: float
    logic: str


DEFAULT_FOLLOW_THRESHOLD = 2.0
US_MARKET = "美股"
KR_MARKET = "韩股"

FOREIGN_SYMBOL_ALIASES = {
    "超微半导体": "AMD",
    "超威半导体": "AMD",
    "ADVANCED MICRO DEVICES": "AMD",
    "AMD.O": "AMD",
    "NASDAQ:AMD": "AMD",
    "US.AMD": "AMD",
    "美光": "MU",
    "美光科技": "MU",
    "MICRON": "MU",
    "MICRON TECHNOLOGY": "MU",
    "MU.O": "MU",
    "NASDAQ:MU": "MU",
    "US.MU": "MU",
}


FOLLOW_TRADE_MAP: list[UpstreamLink] = [
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "601138", "工业富联", "AI服务器代工/组装", 0.92, "海外AI服务器需求上行时，服务器制造和组装环节容易先被资金跟随。"),
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "300308", "中际旭创", "高速光模块", 0.90, "AI集群扩张会带动800G/1.6T光模块需求预期。"),
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "300502", "新易盛", "高速光模块", 0.86, "与AI数据中心光互联景气度相关。"),
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "300394", "天孚通信", "光器件", 0.82, "光模块上游器件环节受AI光互联需求影响。"),
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "300476", "胜宏科技", "AI服务器PCB", 0.82, "AI服务器和加速卡拉动高端PCB需求预期。"),
    UpstreamLink("NVDA", "英伟达", "美股", "AI算力", "002463", "沪电股份", "高速PCB", 0.78, "高速交换机和服务器PCB受算力资本开支预期影响。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "002156", "通富微电", "CPU/GPU封装测试", 0.88, "通富微电与AMD封测链条关联度较高，AMD大涨时容易映射到先进封装和封测方向。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "300308", "中际旭创", "高速光模块", 0.78, "AI加速卡放量预期通常带动数据中心互联链条。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "300502", "新易盛", "高速光模块", 0.74, "AI服务器和GPU集群扩张会带动高速光模块需求预期。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "601138", "工业富联", "AI服务器代工/组装", 0.72, "服务器整机链条可能跟随海外AI芯片订单情绪。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "002463", "沪电股份", "高速PCB", 0.72, "AI服务器、交换机和加速卡需求改善时，高速PCB链条可能受益。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "300476", "胜宏科技", "AI服务器PCB", 0.70, "GPU服务器和加速卡放量预期会映射高端PCB方向。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "600584", "长电科技", "封装测试", 0.66, "AMD走强时会带动先进封装、封测和半导体制造链情绪。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "002185", "华天科技", "封装测试", 0.60, "封测板块可作为海外芯片景气扩散的低强度映射。"),
    UpstreamLink("AMD", "超微半导体", "美股", "AI算力", "688041", "海光信息", "国产CPU/GPGPU映射", 0.56, "AMD上涨有时会带动A股CPU/GPU算力方向的情绪映射，但产业链直接度低于封测和服务器链。"),
    UpstreamLink("TSM", "台积电", "美股", "半导体制造", "002371", "北方华创", "半导体设备", 0.78, "晶圆厂资本开支预期改善时，设备链容易跟随。"),
    UpstreamLink("TSM", "台积电", "美股", "半导体制造", "688012", "中微公司", "刻蚀/MOCVD设备", 0.76, "先进制程和晶圆厂扩产情绪会传导到设备国产替代链。"),
    UpstreamLink("TSM", "台积电", "美股", "半导体制造", "688072", "拓荆科技", "薄膜沉积设备", 0.70, "晶圆制造景气度与薄膜设备需求相关。"),
    UpstreamLink("ASML", "阿斯麦", "美股", "半导体设备", "002371", "北方华创", "半导体设备", 0.82, "全球半导体设备情绪改善时，A股设备龙头通常有映射。"),
    UpstreamLink("ASML", "阿斯麦", "美股", "半导体设备", "688012", "中微公司", "半导体设备", 0.80, "海外设备龙头大涨常映射国产设备链。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "301308", "江波龙", "存储模组", 0.78, "存储价格和景气预期上行时，模组和品牌存储链条容易跟随。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "688525", "佰维存储", "存储模组", 0.75, "存储周期改善预期的A股映射。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "001309", "德明利", "存储控制/模组", 0.72, "存储周期和模组景气改善时，存储控制和模组方向容易跟随。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "300475", "香农芯创", "存储分销/模组链", 0.70, "美光和存储价格上行会带动存储分销、模组及产业链情绪。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "000021", "深科技", "存储封测/制造", 0.68, "存储景气改善时，封测和制造环节可能跟随。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "603986", "兆易创新", "存储芯片/MCU", 0.66, "存储周期和半导体情绪改善时可能联动。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "300223", "北京君正", "存储芯片/车载存储", 0.62, "存储芯片景气改善时，A股存储设计方向可能跟随。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "688766", "普冉股份", "非易失性存储", 0.58, "存储周期改善可扩散到非易失性存储芯片方向。"),
    UpstreamLink("MU", "美光科技", "美股", "存储", "002409", "雅克科技", "半导体材料/存储材料", 0.56, "存储厂景气改善时，上游电子材料和前驱体方向可能获得情绪映射。"),
    UpstreamLink("AAPL", "苹果", "美股", "消费电子", "002475", "立讯精密", "消费电子组装/零组件", 0.88, "苹果链核心公司，隔夜苹果上涨常带来果链情绪映射。"),
    UpstreamLink("AAPL", "苹果", "美股", "消费电子", "002241", "歌尔股份", "声学/智能硬件", 0.76, "苹果及消费电子需求改善预期映射。"),
    UpstreamLink("AAPL", "苹果", "美股", "消费电子", "300433", "蓝思科技", "玻璃/结构件", 0.72, "苹果硬件需求与结构件供应链相关。"),
    UpstreamLink("AAPL", "苹果", "美股", "消费电子", "002938", "鹏鼎控股", "FPC/PCB", 0.72, "苹果终端销量和创新周期影响FPC链条。"),
    UpstreamLink("TSLA", "特斯拉", "美股", "新能源汽车", "300750", "宁德时代", "动力电池", 0.84, "特斯拉销量和降本预期会影响动力电池链情绪。"),
    UpstreamLink("TSLA", "特斯拉", "美股", "新能源汽车", "002050", "三花智控", "热管理", 0.82, "热管理零部件与新能源车产业链高度相关。"),
    UpstreamLink("TSLA", "特斯拉", "美股", "新能源汽车", "601689", "拓普集团", "汽车零部件", 0.78, "特斯拉产业链和智能电动车零部件映射。"),
    UpstreamLink("TSLA", "特斯拉", "美股", "新能源汽车", "002472", "双环传动", "传动系统", 0.62, "新能源车零部件景气跟随。"),
    UpstreamLink("005930.KS", "三星电子", "韩股", "半导体/消费电子", "002371", "北方华创", "半导体设备", 0.72, "三星半导体景气改善会映射到半导体设备和材料链。"),
    UpstreamLink("005930.KS", "三星电子", "韩股", "半导体/消费电子", "301308", "江波龙", "存储模组", 0.70, "三星存储周期变化会影响A股存储链情绪。"),
    UpstreamLink("005930.KS", "三星电子", "韩股", "半导体/消费电子", "002475", "立讯精密", "消费电子零组件", 0.58, "消费电子景气和零组件链可能跟随。"),
    UpstreamLink("000660.KS", "SK海力士", "韩股", "存储/HBM", "301308", "江波龙", "存储模组", 0.82, "HBM和存储涨价预期改善时，存储模组链条容易跟随。"),
    UpstreamLink("000660.KS", "SK海力士", "韩股", "存储/HBM", "688525", "佰维存储", "存储模组", 0.80, "存储周期向上时的A股高弹性映射。"),
    UpstreamLink("000660.KS", "SK海力士", "韩股", "存储/HBM", "603986", "兆易创新", "存储芯片/MCU", 0.66, "存储景气回升带动半导体情绪。"),
    UpstreamLink("373220.KS", "LG新能源", "韩股", "动力电池", "300750", "宁德时代", "动力电池", 0.80, "全球动力电池情绪改善时，A股电池龙头容易跟随。"),
    UpstreamLink("373220.KS", "LG新能源", "韩股", "动力电池", "300014", "亿纬锂能", "动力/储能电池", 0.72, "电池需求和储能景气预期映射。"),
    UpstreamLink("373220.KS", "LG新能源", "韩股", "动力电池", "002812", "恩捷股份", "隔膜", 0.68, "电池上游材料需求预期传导。"),
    UpstreamLink("373220.KS", "LG新能源", "韩股", "动力电池", "002709", "天赐材料", "电解液", 0.66, "电池材料环节跟随动力电池情绪。"),
    UpstreamLink("006400.KS", "三星SDI", "韩股", "动力电池", "300750", "宁德时代", "动力电池", 0.72, "韩股电池龙头上涨常映射A股电池链。"),
    UpstreamLink("006400.KS", "三星SDI", "韩股", "动力电池", "300073", "当升科技", "正极材料", 0.66, "电池扩产和需求改善预期传导到正极材料。"),
    UpstreamLink("005380.KS", "现代汽车", "韩股", "汽车", "601689", "拓普集团", "汽车零部件", 0.62, "全球汽车景气改善时，零部件链条可能跟随。"),
    UpstreamLink("005380.KS", "现代汽车", "韩股", "汽车", "002050", "三花智控", "热管理", 0.60, "汽车热管理链条具备全球车企映射。"),
    UpstreamLink("000270.KS", "起亚", "韩股", "汽车", "601689", "拓普集团", "汽车零部件", 0.58, "汽车销量改善预期映射零部件链。"),
]


def known_foreign_names() -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for link in FOLLOW_TRADE_MAP:
        key = link.foreign_symbol
        if key not in seen:
            seen.add(key)
            names.append(f"{link.market} | {link.foreign_symbol} | {link.foreign_name}")
    return names


def foreign_watchlist(markets: list[str] | None = None) -> pd.DataFrame:
    allowed = set(markets or [US_MARKET, KR_MARKET])
    rows: dict[str, dict] = {}
    for link in FOLLOW_TRADE_MAP:
        if link.market not in allowed:
            continue
        rows.setdefault(
            link.foreign_symbol,
            {
                "选择": False,
                "市场": link.market,
                "代码": link.foreign_symbol,
                "名称": link.foreign_name,
                "产业主题": link.sector,
                "最近涨幅%": 0.0,
                "前一交易日涨跌幅%": 0.0,
                "映射A股数": 0,
                "数据状态": "待获取",
            },
        )
        rows[link.foreign_symbol]["映射A股数"] += 1
    return pd.DataFrame(rows.values()).sort_values(["市场", "代码"]).reset_index(drop=True)


def build_foreign_strength_pool(
    markets: list[str] | None = None,
    lookback_days: int = 5,
    min_recent_pct: float = 0.0,
    max_symbols: int = 80,
) -> pd.DataFrame:
    pool = foreign_watchlist(markets)
    if pool.empty:
        return pool

    rows: list[dict] = []
    for _, item in pool.head(max_symbols).iterrows():
        row = item.to_dict()
        if row["市场"] == US_MARKET:
            try:
                recent_pct, last_pct = _us_recent_returns(str(row["代码"]), lookback_days)
                row["最近涨幅%"] = round(recent_pct, 2)
                row["前一交易日涨跌幅%"] = round(last_pct, 2)
                row["数据状态"] = "自动获取"
            except Exception as exc:
                row["数据状态"] = f"获取失败：{str(exc)[:40]}"
        elif row["市场"] == KR_MARKET:
            row["数据状态"] = "韩股暂用观察池，可手动改涨跌幅"
        rows.append(row)

    output = pd.DataFrame(rows)
    numeric = pd.to_numeric(output["最近涨幅%"], errors="coerce").fillna(0)
    keep_fallback = output["市场"] == KR_MARKET
    output = output[(numeric >= min_recent_pct) | keep_fallback].copy()
    return output.sort_values(["最近涨幅%", "前一交易日涨跌幅%", "映射A股数"], ascending=False).reset_index(drop=True)


def selected_moves_from_pool(frame: pd.DataFrame, pct_column: str = "前一交易日涨跌幅%") -> list[ForeignMove]:
    if frame.empty or "选择" not in frame.columns:
        return []
    selected = frame[frame["选择"].fillna(False).astype(bool)]
    moves: list[ForeignMove] = []
    for _, row in selected.iterrows():
        symbol = _normalize_foreign_symbol(str(row.get("代码", "")))
        if not symbol:
            continue
        try:
            pct_change = float(str(row.get(pct_column, 0)).replace("%", ""))
        except ValueError:
            pct_change = 0.0
        moves.append(
            ForeignMove(
                symbol=symbol,
                name=str(row.get("名称", "")) or symbol,
                market=str(row.get("市场", "")) or "海外",
                pct_change=pct_change,
            )
        )
    return moves


def parse_foreign_moves(text: str) -> list[ForeignMove]:
    moves: list[ForeignMove] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in re.split(r"[,，\t| ]+", line) if part.strip()]
        if len(parts) < 2:
            continue
        pct_text = parts[-1].replace("%", "")
        try:
            pct_change = float(pct_text)
        except ValueError:
            continue
        symbol = _normalize_foreign_symbol(parts[0])
        matched = _links_for_symbol(symbol)
        name = matched[0].foreign_name if matched else symbol
        market = matched[0].market if matched else "海外"
        moves.append(ForeignMove(symbol=symbol, name=name, market=market, pct_change=pct_change))
    return moves


def moves_from_frame(frame: pd.DataFrame) -> list[ForeignMove]:
    moves: list[ForeignMove] = []
    if frame.empty:
        return moves
    for _, row in frame.iterrows():
        symbol = _normalize_foreign_symbol(str(row.get("代码", row.get("symbol", ""))))
        if not symbol:
            continue
        try:
            pct_change = float(str(row.get("昨夜涨跌幅%", row.get("pct_change", 0))).replace("%", ""))
        except ValueError:
            continue
        links = _links_for_symbol(symbol)
        moves.append(
            ForeignMove(
                symbol=symbol,
                name=str(row.get("名称", "")) or (links[0].foreign_name if links else symbol),
                market=str(row.get("市场", "")) or (links[0].market if links else "海外"),
                pct_change=pct_change,
            )
        )
    return moves


def build_follow_candidates(
    moves: list[ForeignMove],
    min_up_pct: float = DEFAULT_FOLLOW_THRESHOLD,
    include_negative: bool = False,
) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for move in moves:
        if move.pct_change < min_up_pct and not include_negative:
            continue
        if abs(move.pct_change) < min_up_pct:
            continue
        for link in _links_for_symbol(move.symbol):
            direction = 1 if move.pct_change >= 0 else -1
            score = abs(move.pct_change) * 20 * link.relation_strength
            action = "跟随观察/开盘买入候选" if direction > 0 else "负向冲击/回避观察"
            current = rows.setdefault(
                link.ashare_symbol,
                {
                    "A股代码": link.ashare_symbol,
                    "A股名称": link.ashare_name,
                    "产业链环节": link.upstream_role,
                    "触发源": [],
                    "海外市场": [],
                    "昨夜涨跌幅%": [],
                    "映射强度": [],
                    "跟随分": 0.0,
                    "操作方向": action,
                    "逻辑": [],
                    "风控提示": "仅适合开盘前计划。若A股高开超过5%不追，跌破开盘价且量能放大则放弃。",
                },
            )
            current["触发源"].append(f"{move.name}({move.symbol})")
            current["海外市场"].append(move.market)
            current["昨夜涨跌幅%"].append(move.pct_change)
            current["映射强度"].append(link.relation_strength)
            current["跟随分"] += score * direction
            current["逻辑"].append(link.logic)
            if direction < 0:
                current["操作方向"] = "负向冲击/回避观察"

    output = []
    for row in rows.values():
        pct_values = row["昨夜涨跌幅%"]
        strengths = row["映射强度"]
        output.append(
            {
                "A股代码": row["A股代码"],
                "A股名称": row["A股名称"],
                "产业链环节": row["产业链环节"],
                "触发源": "、".join(dict.fromkeys(row["触发源"])),
                "海外市场": "、".join(dict.fromkeys(row["海外市场"])),
                "昨夜最大涨跌幅%": round(max(pct_values, key=abs), 2),
                "平均映射强度": round(sum(strengths) / len(strengths), 2),
                "跟随分": round(row["跟随分"], 2),
                "操作方向": row["操作方向"],
                "逻辑": "；".join(dict.fromkeys(row["逻辑"])),
                "风控提示": row["风控提示"],
            }
        )
    if not output:
        return pd.DataFrame()
    return pd.DataFrame(output).sort_values("跟随分", ascending=False).reset_index(drop=True)


def sample_moves_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"市场": "美股", "代码": "NVDA", "名称": "英伟达", "昨夜涨跌幅%": 3.5},
            {"市场": "韩股", "代码": "000660.KS", "名称": "SK海力士", "昨夜涨跌幅%": 2.8},
            {"市场": "美股", "代码": "TSLA", "名称": "特斯拉", "昨夜涨跌幅%": 1.2},
        ]
    )


def _links_for_symbol(symbol: str) -> list[UpstreamLink]:
    normalized = _normalize_foreign_symbol(symbol)
    return [link for link in FOLLOW_TRADE_MAP if link.foreign_symbol.upper() == normalized]


def _normalize_foreign_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized in FOREIGN_SYMBOL_ALIASES:
        return FOREIGN_SYMBOL_ALIASES[normalized]
    if normalized.startswith(("NASDAQ:", "NYSE:")):
        normalized = normalized.split(":", 1)[1]
    if normalized.startswith("US."):
        normalized = normalized.split(".", 1)[1]
    if normalized.endswith(".O"):
        normalized = normalized[:-2]
    return FOREIGN_SYMBOL_ALIASES.get(normalized, normalized)


def _us_recent_returns(symbol: str, lookback_days: int) -> tuple[float, float]:
    import akshare as ak

    raw = ak.stock_us_daily(symbol=symbol, adjust="")
    if raw.empty or len(raw) < 2:
        raise ValueError("no US daily data")
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("not enough US daily data")
    window = max(2, min(int(lookback_days), len(data) - 1))
    recent = float(data["close"].iloc[-1] / data["close"].iloc[-window] - 1) * 100
    last = float(data["close"].iloc[-1] / data["close"].iloc[-2] - 1) * 100
    return recent, last
