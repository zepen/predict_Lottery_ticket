# 澳门六合彩分析系统

一个完整的澳门六合彩数据分析系统，具备数据爬取、深度分析、预测推荐和可视化功能。

## 功能特点

### 🔍 数据获取
- 自动爬取澳门六合彩001-289期开奖数据
- 支持多种数据源和备用方案
- 自动生成测试数据用于系统测试
- 数据持久化存储（SQLite数据库）

### 📊 深度分析
- **频率统计**：计算每个号码的出现频率和热度分数
- **热门冷门分析**：识别热门和冷门号码
- **模式分析**：分析连续号码对、奇偶比例、号码和分布等
- **交替矩阵**：分析号码间的关联性
- **深度思考模式**：多维度综合分析

### 🎯 预测推荐
- 基于热度分数的推荐
- 基于模式分析的推荐
- 基于交替矩阵的推荐
- 综合推荐算法
- 置信度评估

### 📈 可视化报告
- 频率分析图表
- 热门冷门号码图表
- 模式分析图表
- 交替矩阵热力图
- 交互式仪表板
- 综合分析报告（HTML格式）

### 🔄 自动化功能
- 定时自动更新数据
- 自动生成分析报告
- 历史记录保存
- 增量数据分析

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
# 运行完整分析（使用测试数据）
python main.py --test

# 运行完整分析（爬取真实数据）
python main.py --start 1 --end 289

# 仅爬取数据
python main.py --crawl-only --start 1 --end 289

# 仅分析已有数据
python main.py --analyze-only --start 1 --end 289

# 启动自动更新服务
python main.py --auto-update
```

### 高级使用

```python
from main import MacauLotterySystem

# 创建系统实例
system = MacauLotterySystem()

# 生成测试数据
system.generate_test_data(1, 289)

# 运行分析
analysis_result = system.run_complete_analysis(1, 289, use_test_data=True)

# 查看结果
print("推荐号码:", analysis_result['recommendations']['comprehensive'])
print("置信度:", analysis_result['confidence_score'])
```

## 系统架构

```
澳门六合彩分析系统/
├── main.py              # 主程序入口
├── crawler.py           # 数据爬虫模块
├── analyzer.py          # 数据分析模块
├── visualizer.py        # 可视化模块
├── database.py          # 数据库操作模块
├── config.py            # 配置文件
├── requirements.txt     # 依赖包列表
├── data/               # 数据存储目录
├── reports/            # 报告输出目录
└── logs/              # 日志目录
```

## 分析算法

### 1. 频率分析
- 计算每个号码的出现频率
- 基于频率和最近出现情况计算热度分数
- 识别热门号码（热度分数 ≥ 0.6）
- 识别冷门号码（冷度分数 ≥ 0.3）

### 2. 模式分析
- **连续号码对**：分析相邻号码的出现模式
- **号码和分布**：分析开奖号码和的范围和分布
- **奇偶比例**：分析奇数和偶数的分布规律
- **号码间隔**：分析号码间的间隔分布

### 3. 交替矩阵
- 构建49x49的号码关联矩阵
- 分析号码间的共同出现频率
- 识别强关联的号码组合

### 4. 综合推荐
- 结合热度分数、模式分析和交替矩阵
- 计算综合推荐分数
- 生成最终推荐号码列表
- 提供置信度评估

## 数据格式

### 开奖记录
- 期数（period）
- 开奖日期（draw_date）
- 开奖号码（numbers，逗号分隔）
- 特别号码（special_number）

### 分析结果
- 热门号码列表
- 冷门号码列表
- 推荐号码列表
- 置信度分数
- 详细统计信息

## 输出文件

- `data/macau_lottery_*.xlsx`：Excel格式的开奖数据
- `reports/frequency_analysis_*.png`：频率分析图表
- `reports/hot_cold_numbers_*.png`：热门冷门号码图表
- `reports/pattern_analysis_*.png`：模式分析图表
- `reports/interactive_dashboard_*.html`：交互式仪表板
- `reports/comprehensive_report_*.html`：综合分析报告

## 注意事项

1. **数据来源**：系统优先尝试爬取真实数据，失败时自动生成测试数据
2. **分析准确性**：预测结果仅供参考，不保证准确性
3. **数据更新**：建议定期更新数据以获得最新分析结果
4. **系统要求**：需要Python 3.7+和稳定的网络连接

## 技术栈

- **Python 3.7+**
- **数据爬取**：requests, beautifulsoup4, selenium
- **数据分析**：pandas, numpy
- **可视化**：matplotlib, seaborn, plotly
- **数据库**：sqlite3
- **定时任务**：schedule

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持数据爬取和分析
- 实现基础推荐算法
- 提供可视化功能