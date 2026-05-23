# stockbuyornot

A股量价信号解释器与可视化工作台。程序把日线 OHLCV 数据转换成结构阶段、关键支撑/压力、量价信号、评分、止损与解释，用于研究、复盘和候选股筛选。

> 本项目仅用于研究和辅助复盘，不构成投资建议。真实交易前必须做历史回测、参数校准、滑点/手续费建模和风控验证。

## 打开可视化工作台

```powershell
cd C:\Users\pro6a\Documents\stockbuyornot
.\workbench.cmd
```

浏览器打开：

```text
http://127.0.0.1:8501
```

如果页面打不开或提示 `127.0.0.1 拒绝连接`，运行：

```powershell
.\restart_workbench.cmd
```

## 命令行使用

分析单票：

```powershell
.\stockbuyornot.cmd analyze --symbol 000001 --start 20240101 --end 20260517
```

扫描股票池：

```powershell
.\stockbuyornot.cmd scan --symbols 000001 600519 300750 --start 20240101 --end 20260517 --min-score 60
```

回测：

```powershell
.\stockbuyornot.cmd backtest --symbol 000001 --start 20200101 --end 20260517
```

更多说明见 `docs/USAGE.md`。

完整产品说明书见 `docs/PRODUCT_MANUAL.md`，包含产品定位、量价算法、强股雷达、名词解释、预测模型、参数说明和上线风险提示。

## 自检

```powershell
.\check.ps1
```
