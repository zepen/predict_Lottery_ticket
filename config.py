# 澳门六合彩分析系统配置文件
import os

# 数据库配置
DATABASE_PATH = "macau_lottery.db"

# 网站配置
BASE_URL = "https://kj.123720c.com"
LOTTERY_URL = "https://kj.123720c.com/kj/"

# 六合彩号码范围
NUMBER_RANGE = list(range(1, 50))  # 1-49号
TOTAL_NUMBERS = 49

# 分析配置
HOT_THRESHOLD = 0.6  # 热门号码阈值
COLD_THRESHOLD = 0.3  # 冷门号码阈值
ANALYSIS_PERIODS = 288  # 分析期数

# 文件路径
DATA_DIR = "data"
REPORTS_DIR = "reports"
LOGS_DIR = "logs"

# 创建必要目录
for directory in [DATA_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)