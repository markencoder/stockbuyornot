# 使用说明

## 当前环境

程序已安装在：

```text
C:\ProgramData\Anaconda3\envs\tower312
```

推荐在项目目录运行：

```powershell
cd C:\Users\pro6a\Documents\stockbuyornot
```

## 打开可视化工作台

```powershell
.\workbench.cmd
```

然后打开：

```text
http://127.0.0.1:8501
```

如果打不开，或浏览器提示 `127.0.0.1 拒绝连接`，运行：

```powershell
.\restart_workbench.cmd
```

## 工作台功能

- 单票诊断
- 股票池扫描
- 策略回测
- 跟随交易
- 数据工具
- 使用说明

## 命令行分析

```powershell
.\stockbuyornot.cmd analyze --symbol 000001 --start 20240101 --end 20260517
```

## 命令行扫描

```powershell
.\stockbuyornot.cmd scan --symbols 000001 600519 300750 --start 20240101 --end 20260517 --min-score 60
```

导出结果：

```powershell
.\stockbuyornot.cmd scan --symbols 000001 600519 300750 --start 20240101 --end 20260517 --min-score 60 --output candidates.csv
```

## 本地 CSV

单个文件：

```powershell
.\stockbuyornot.cmd analyze --csv data\daily\000001.csv
```

文件夹扫描：

```powershell
.\stockbuyornot.cmd scan --csv-dir data\daily --start 20240101 --end 20260517 --min-score 70 --output candidates.csv
```

CSV 至少包含：

```text
date,open,high,low,close,volume
```

推荐包含：

```text
amount,symbol
```

## 回测

```powershell
.\stockbuyornot.cmd backtest --symbol 000001 --start 20200101 --end 20260517
```

## 跟随交易

工作台新增“跟随交易”页，用于开盘前根据美股、韩股前一晚涨跌，映射A股上游企业。

使用方式：

```text
1. 选择“自动选强势股”，刷新近期美股/韩股观察池。
2. 勾选要跟随的海外标的；美股会自动计算近期涨幅和前一交易日涨跌幅。
3. 韩股当前作为重点观察池展示，可在表格里手动补充涨跌幅。
4. 选择用“前一交易日涨跌幅”或“最近涨幅”生成跟随信号。
5. 程序根据内置产业链映射生成A股候选。
6. 可勾选“叠加A股量价诊断”，过滤掉量价结构较弱的标的。
7. 下载 follow_trade_candidates.csv 作为开盘观察清单。
```

注意：

```text
跟随交易只生成观察清单，不自动下单。
若A股高开超过5%不追；开盘后跌破开盘价且放量，应放弃跟随。
映射表是研究辅助，后续需要定期维护产业链关系。
```

## 组合回测：趋势回踩策略

这个入口只做“第二阶段主升 + 缩量回踩 + 上涨中继买点”，适合用来评估股票池级别的实际收益，不再把突破、底部反转、区间低吸混在一起。

小股票池测试：

```powershell
.\stockbuyornot.cmd portfolio-backtest --symbols 000001 600519 300750 --start 20240101 --end 20260517 --output candidate\trend_pullback_test.csv
```

指数股票池：

```powershell
.\stockbuyornot.cmd portfolio-backtest --pool sse50 --start 20240101 --end 20260517 --max-positions 5 --output candidate\sse50_trend_pullback.csv
.\stockbuyornot.cmd portfolio-backtest --pool csi300 --start 20240101 --end 20260517 --max-positions 5 --output candidate\csi300_trend_pullback.csv
.\stockbuyornot.cmd portfolio-backtest --pool csi500 --start 20240101 --end 20260517 --max-positions 5 --output candidate\csi500_trend_pullback.csv
.\stockbuyornot.cmd portfolio-backtest --pool chinext --start 20240101 --end 20260517 --max-positions 5 --output candidate\chinext_trend_pullback.csv
```

先做 smoke test，避免一次跑太久：

```powershell
.\stockbuyornot.cmd portfolio-backtest --pool sse50 --limit 20 --start 20240101 --end 20260517 --output candidate\sse50_limit20.csv
```

输出会生成多份文件：

```text
*_equity.csv       每日净值
*_trades.csv       交易记录
*_positions.csv    每日持仓
*_candidates.csv   每日候选排名
*_metrics.json     收益、回撤、胜率等指标
```

常用参数：

```powershell
--market-mode balanced          强市开满仓，震荡市最多开少量仓，弱市不开新仓
--max-positions 5               强市最多持仓数
--neutral-max-positions 2       震荡市最多持仓数
--min-score 75                  入选最低解释分
--min-avg-amount 50000000       近20日最低平均成交额
--max-stop-distance 0.07        最大止损距离
--min-relative-strength 0.03    60日相对基准最低强度
--min-reward-risk 1.8           最低预估盈亏比
--breakeven-r 1.0               盈利达到1R后移动到保本止损
--trail-start-r 2.0             盈利达到2R后启用趋势跟踪止盈
--trail-pct 0.10                最高收盘价回撤止盈阈值
--stale-days 12                 12个交易日未达到1R则退出
--max-holding-days 45           最长持仓交易日
```

默认基准会随股票池选择：

```text
sse50   -> 000016
csi300  -> 000300
csi500  -> 000905
chinext -> 399006
其他    -> 000300
```

## 前端回测页

工作台的“策略回测”页已经同步支持以下参数：

- 最低买入分
- 初始资金
- 单笔风险比例
- 佣金率
- 印花税
- 滑点

回测结果会展示：

- 初始资金、期末资金、总收益
- 交易次数、胜率、平均单笔收益
- 资金曲线
- 交易流水
- 配对盈亏
- 回测规则说明
