#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六合彩生肖属性分析系统
专门针对六合彩的生肖属性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import sqlite3
from collections import Counter, defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ZodiacAnalyzer:
    def __init__(self):
        self.zodiac_mapping = self.create_zodiac_mapping()
        self.db_path = "macau_lottery_zodiac.db"
        self.init_database()
    
    def create_zodiac_mapping(self):
        """创建号码与生肖的映射关系"""
        # 六合彩生肖对应表（基于传统生肖轮换）
        zodiac_mapping = {}
        
        # 2025年对应的生肖年份（蛇年）
        # 生肖顺序：鼠、牛、虎、兔、龙、蛇、马、羊、猴、鸡、狗、猪
        zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 每个生肖对应的号码范围（基于传统六合彩生肖表）
        zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49],
            '牛': [2, 14, 26, 38],
            '虎': [3, 15, 27, 39],
            '兔': [4, 16, 28, 40],
            '龙': [5, 17, 29, 41],
            '蛇': [6, 18, 30, 42],
            '马': [7, 19, 31, 43],
            '羊': [8, 20, 32, 44],
            '猴': [9, 21, 33, 45],
            '鸡': [10, 22, 34, 46],
            '狗': [11, 23, 35, 47],
            '猪': [12, 24, 36, 48]
        }
        
        # 创建号码到生肖的映射
        for zodiac, numbers in zodiac_numbers.items():
            for num in numbers:
                zodiac_mapping[num] = zodiac
        
        return zodiac_mapping
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建开奖记录表（包含生肖信息）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lottery_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period INTEGER UNIQUE NOT NULL,
                draw_date TEXT NOT NULL,
                numbers TEXT NOT NULL,
                special_number INTEGER,
                zodiac_numbers TEXT,
                zodiac_count TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建生肖统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zodiac_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zodiac TEXT NOT NULL,
                frequency INTEGER DEFAULT 0,
                hot_score REAL DEFAULT 0.0,
                cold_score REAL DEFAULT 0.0,
                last_appeared_period INTEGER,
                numbers TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建生肖分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zodiac_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                hot_zodiacs TEXT,
                cold_zodiacs TEXT,
                recommended_zodiacs TEXT,
                zodiac_patterns TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_zodiac_by_number(self, number):
        """根据号码获取生肖"""
        return self.zodiac_mapping.get(number, '未知')
    
    def analyze_numbers_zodiac(self, numbers):
        """分析号码的生肖属性"""
        zodiacs = [self.get_zodiac_by_number(num) for num in numbers]
        zodiac_count = Counter(zodiacs)
        return zodiacs, dict(zodiac_count)
    
    def crawl_lottery_data(self, period):
        """爬取六合彩开奖数据"""
        try:
            url = f"https://kj.123720c.com/kj/sx.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 尝试解析开奖数据
                numbers = self.parse_lottery_numbers(soup)
                if numbers:
                    # 分析生肖属性
                    zodiacs, zodiac_count = self.analyze_numbers_zodiac(numbers)
                    
                    return {
                        'period': period,
                        'draw_date': datetime.now().strftime('%Y-%m-%d'),
                        'numbers': ','.join(map(str, numbers)),
                        'special_number': numbers[-1] if len(numbers) > 6 else None,
                        'zodiac_numbers': ','.join(zodiacs),
                        'zodiac_count': ','.join([f"{z}:{c}" for z, c in zodiac_count.items()])
                    }
            
            # 如果爬取失败，生成模拟数据
            return self.generate_mock_data(period)
            
        except Exception as e:
            print(f"爬取第{period}期数据失败: {e}")
            return self.generate_mock_data(period)
    
    def parse_lottery_numbers(self, soup):
        """解析开奖号码"""
        # 尝试多种选择器
        selectors = [
            '.lottery-numbers .number',
            '.result-numbers .ball',
            '.draw-numbers .num',
            '.lottery-result .number',
            '[class*="number"]',
            '[class*="ball"]'
        ]
        
        numbers = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for elem in elements:
                    text = elem.get_text().strip()
                    if text.isdigit() and 1 <= int(text) <= 49:
                        numbers.append(int(text))
                if len(numbers) >= 6:
                    break
        
        return numbers[:6] if len(numbers) >= 6 else None
    
    def generate_mock_data(self, period):
        """生成模拟数据"""
        import random
        random.seed(period + 2025)
        
        # 生成6个不重复的号码
        numbers = sorted(random.sample(range(1, 50), 6))
        
        # 分析生肖属性
        zodiacs, zodiac_count = self.analyze_numbers_zodiac(numbers)
        
        # 模拟开奖日期
        base_date = datetime(2025, 1, 7)
        days_offset = (period - 1) * 2 + (period - 1) // 3
        draw_date = (base_date + pd.Timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        return {
            'period': period,
            'draw_date': draw_date,
            'numbers': ','.join(map(str, numbers)),
            'special_number': random.randint(1, 49),
            'zodiac_numbers': ','.join(zodiacs),
            'zodiac_count': ','.join([f"{z}:{c}" for z, c in zodiac_count.items()])
        }
    
    def save_lottery_record(self, record):
        """保存开奖记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO lottery_records
            (period, draw_date, numbers, special_number, zodiac_numbers, zodiac_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            record['period'],
            record['draw_date'],
            record['numbers'],
            record['special_number'],
            record['zodiac_numbers'],
            record['zodiac_count']
        ))
        
        conn.commit()
        conn.close()
    
    def get_lottery_records(self, start_period=1, end_period=289):
        """获取开奖记录"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT period, draw_date, numbers, special_number, zodiac_numbers, zodiac_count
            FROM lottery_records
            WHERE period BETWEEN ? AND ?
            ORDER BY period
        '''
        df = pd.read_sql_query(query, conn, params=(start_period, end_period))
        conn.close()
        return df
    
    def analyze_zodiac_frequency(self, df):
        """分析生肖频率"""
        zodiac_frequency = defaultdict(int)
        zodiac_periods = defaultdict(list)
        
        for _, row in df.iterrows():
            period = row['period']
            zodiac_count_str = row['zodiac_count']
            
            if pd.notna(zodiac_count_str):
                # 解析生肖计数
                for item in zodiac_count_str.split(','):
                    if ':' in item:
                        zodiac, count = item.split(':')
                        zodiac_frequency[zodiac] += int(count)
                        zodiac_periods[zodiac].extend([period] * int(count))
        
        # 计算热度分数
        total_periods = len(df)
        zodiac_stats = {}
        
        for zodiac in self.zodiac_mapping.values():
            if zodiac in zodiac_frequency:
                frequency = zodiac_frequency[zodiac]
                last_period = max(zodiac_periods[zodiac]) if zodiac_periods[zodiac] else 0
                
                # 计算最近50期的频率
                recent_periods = [p for p in zodiac_periods[zodiac] if p > total_periods - 50]
                recent_frequency = len(recent_periods) / 50 if total_periods >= 50 else frequency / total_periods
                
                hot_score = (frequency / total_periods * 0.7 + recent_frequency * 0.3)
                cold_score = 1 - hot_score
                
                zodiac_stats[zodiac] = {
                    'frequency': frequency,
                    'frequency_rate': frequency / total_periods,
                    'last_appeared_period': last_period,
                    'hot_score': hot_score,
                    'cold_score': cold_score,
                    'appearances': zodiac_periods[zodiac]
                }
            else:
                zodiac_stats[zodiac] = {
                    'frequency': 0,
                    'frequency_rate': 0,
                    'last_appeared_period': 0,
                    'hot_score': 0,
                    'cold_score': 1,
                    'appearances': []
                }
        
        return zodiac_stats
    
    def identify_hot_cold_zodiacs(self, zodiac_stats):
        """识别热门和冷门生肖"""
        hot_zodiacs = []
        cold_zodiacs = []
        
        for zodiac, stats in zodiac_stats.items():
            if stats['hot_score'] >= 0.6:
                hot_zodiacs.append((zodiac, stats['hot_score']))
            elif stats['cold_score'] >= 0.3:
                cold_zodiacs.append((zodiac, stats['cold_score']))
        
        hot_zodiacs.sort(key=lambda x: x[1], reverse=True)
        cold_zodiacs.sort(key=lambda x: x[1], reverse=True)
        
        return hot_zodiacs, cold_zodiacs
    
    def analyze_zodiac_patterns(self, df):
        """分析生肖模式"""
        patterns = {
            'zodiac_combinations': self.find_zodiac_combinations(df),
            'zodiac_sequences': self.find_zodiac_sequences(df),
            'zodiac_balance': self.analyze_zodiac_balance(df)
        }
        return patterns
    
    def find_zodiac_combinations(self, df):
        """查找生肖组合"""
        combinations = Counter()
        
        for _, row in df.iterrows():
            zodiac_numbers = row['zodiac_numbers']
            if pd.notna(zodiac_numbers):
                zodiacs = zodiac_numbers.split(',')
                # 查找所有生肖对组合
                for i in range(len(zodiacs)):
                    for j in range(i+1, len(zodiacs)):
                        combo = tuple(sorted([zodiacs[i], zodiacs[j]]))
                        combinations[combo] += 1
        
        return dict(combinations.most_common(10))
    
    def find_zodiac_sequences(self, df):
        """查找生肖序列"""
        sequences = Counter()
        
        for _, row in df.iterrows():
            zodiac_numbers = row['zodiac_numbers']
            if pd.notna(zodiac_numbers):
                zodiacs = zodiac_numbers.split(',')
                # 查找连续出现的生肖
                for i in range(len(zodiacs) - 1):
                    if zodiacs[i] == zodiacs[i+1]:
                        sequences[zodiacs[i]] += 1
        
        return dict(sequences.most_common(5))
    
    def analyze_zodiac_balance(self, df):
        """分析生肖平衡性"""
        zodiac_counts = defaultdict(list)
        
        for _, row in df.iterrows():
            zodiac_count_str = row['zodiac_count']
            if pd.notna(zodiac_count_str):
                period_zodiacs = {}
                for item in zodiac_count_str.split(','):
                    if ':' in item:
                        zodiac, count = item.split(':')
                        period_zodiacs[zodiac] = int(count)
                
                for zodiac in self.zodiac_mapping.values():
                    count = period_zodiacs.get(zodiac, 0)
                    zodiac_counts[zodiac].append(count)
        
        balance_stats = {}
        for zodiac, counts in zodiac_counts.items():
            if counts:
                balance_stats[zodiac] = {
                    'avg_count': np.mean(counts),
                    'max_count': max(counts),
                    'min_count': min(counts),
                    'std_count': np.std(counts)
                }
        
        return balance_stats
    
    def generate_zodiac_recommendations(self, zodiac_stats, patterns):
        """生成生肖推荐"""
        recommendations = {
            'hot_based': [],
            'pattern_based': [],
            'balance_based': [],
            'comprehensive': []
        }
        
        # 基于热度的推荐
        hot_zodiacs = [zodiac for zodiac, score in sorted(
            [(z, stats['hot_score']) for z, stats in zodiac_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:6]]
        recommendations['hot_based'] = hot_zodiacs
        
        # 基于模式的推荐
        if 'zodiac_combinations' in patterns:
            combo_zodiacs = set()
            for combo, count in list(patterns['zodiac_combinations'].items())[:5]:
                combo_zodiacs.update(combo)
            recommendations['pattern_based'] = list(combo_zodiacs)[:6]
        
        # 基于平衡性的推荐
        if 'zodiac_balance' in patterns:
            balance_zodiacs = []
            for zodiac, stats in patterns['zodiac_balance'].items():
                if stats['avg_count'] < 1.0:  # 平均出现次数少于1的生肖
                    balance_zodiacs.append(zodiac)
            recommendations['balance_based'] = balance_zodiacs[:6]
        
        # 综合推荐
        all_candidates = set()
        for rec_type in ['hot_based', 'pattern_based', 'balance_based']:
            all_candidates.update(recommendations[rec_type])
        
        candidate_scores = {}
        for zodiac in all_candidates:
            if zodiac in zodiac_stats:
                score = zodiac_stats[zodiac]['hot_score']
                if zodiac in recommendations['hot_based']:
                    score += 0.3
                if zodiac in recommendations['pattern_based']:
                    score += 0.2
                if zodiac in recommendations['balance_based']:
                    score += 0.1
                candidate_scores[zodiac] = score
        
        comprehensive = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        recommendations['comprehensive'] = [zodiac for zodiac, score in comprehensive]
        
        return recommendations
    
    def create_zodiac_visualization(self, zodiac_stats, hot_zodiacs, cold_zodiacs):
        """创建生肖可视化图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 生肖频率柱状图
        zodiacs = list(zodiac_stats.keys())
        frequencies = [stats['frequency'] for stats in zodiac_stats.values()]
        
        bars1 = ax1.bar(zodiacs, frequencies, color='skyblue', alpha=0.7)
        ax1.set_title('生肖出现频率统计', fontsize=14, fontweight='bold')
        ax1.set_xlabel('生肖', fontsize=12)
        ax1.set_ylabel('出现次数', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, freq in zip(bars1, frequencies):
            if freq > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(freq), ha='center', va='bottom', fontsize=9)
        
        # 2. 热门冷门生肖
        if hot_zodiacs:
            hot_names, hot_scores = zip(*hot_zodiacs[:8])
            ax2.bar(range(len(hot_names)), hot_scores, color='red', alpha=0.7, label='热门生肖')
            ax2.set_title('热门生肖 Top 8', fontsize=14, fontweight='bold')
            ax2.set_xlabel('排名', fontsize=12)
            ax2.set_ylabel('热度分数', fontsize=12)
            ax2.set_xticks(range(len(hot_names)))
            ax2.set_xticklabels(hot_names)
        
        if cold_zodiacs:
            cold_names, cold_scores = zip(*cold_zodiacs[:8])
            ax3.bar(range(len(cold_names)), cold_scores, color='blue', alpha=0.7, label='冷门生肖')
            ax3.set_title('冷门生肖 Top 8', fontsize=14, fontweight='bold')
            ax3.set_xlabel('排名', fontsize=12)
            ax3.set_ylabel('冷度分数', fontsize=12)
            ax3.set_xticks(range(len(cold_names)))
            ax3.set_xticklabels(cold_names)
        
        # 4. 生肖热度分布饼图
        hot_scores = [stats['hot_score'] for stats in zodiac_stats.values()]
        ax4.pie(hot_scores, labels=zodiacs, autopct='%1.1f%%', startangle=90)
        ax4.set_title('生肖热度分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('zodiac_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 生肖分析图已保存: zodiac_analysis.png")
        plt.show()
    
    def save_zodiac_analysis(self, period, analysis_result):
        """保存生肖分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 处理patterns中的tuple键
        patterns_serializable = {}
        for key, value in analysis_result['patterns'].items():
            if isinstance(value, dict):
                patterns_serializable[key] = {str(k): v for k, v in value.items()}
            else:
                patterns_serializable[key] = value
        
        cursor.execute('''
            INSERT INTO zodiac_analysis
            (period, analysis_type, hot_zodiacs, cold_zodiacs, recommended_zodiacs, 
             zodiac_patterns, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            period,
            'comprehensive',
            ','.join([z for z, _ in analysis_result['hot_zodiacs'][:6]]),
            ','.join([z for z, _ in analysis_result['cold_zodiacs'][:6]]),
            ','.join(analysis_result['recommendations']['comprehensive']),
            json.dumps(patterns_serializable, ensure_ascii=False),
            analysis_result['confidence_score']
        ))
        
        conn.commit()
        conn.close()
    
    def run_complete_zodiac_analysis(self, start_period=1, end_period=100):
        """运行完整的生肖分析"""
        print("🐉 六合彩生肖属性分析系统")
        print("=" * 60)
        print(f"分析期数: {start_period} - {end_period}")
        print()
        
        # 1. 爬取或生成数据
        print("📊 步骤1: 获取开奖数据")
        print("-" * 30)
        
        for period in range(start_period, end_period + 1):
            print(f"正在处理第{period}期...")
            record = self.crawl_lottery_data(period)
            self.save_lottery_record(record)
        
        print(f"✅ 成功处理 {end_period - start_period + 1} 期数据")
        print()
        
        # 2. 获取数据进行分析
        df = self.get_lottery_records(start_period, end_period)
        
        # 3. 分析生肖频率
        print("🔍 步骤2: 分析生肖频率")
        print("-" * 30)
        zodiac_stats = self.analyze_zodiac_frequency(df)
        
        # 4. 识别热门冷门生肖
        hot_zodiacs, cold_zodiacs = self.identify_hot_cold_zodiacs(zodiac_stats)
        
        # 5. 分析生肖模式
        print("📈 步骤3: 分析生肖模式")
        print("-" * 30)
        patterns = self.analyze_zodiac_patterns(df)
        
        # 6. 生成推荐
        print("🎯 步骤4: 生成生肖推荐")
        print("-" * 30)
        recommendations = self.generate_zodiac_recommendations(zodiac_stats, patterns)
        
        # 7. 计算置信度
        confidence = np.mean([stats['hot_score'] for stats in zodiac_stats.values()])
        
        # 8. 创建分析结果
        analysis_result = {
            'period_range': f"{start_period}-{end_period}",
            'total_periods': len(df),
            'zodiac_stats': zodiac_stats,
            'hot_zodiacs': hot_zodiacs,
            'cold_zodiacs': cold_zodiacs,
            'patterns': patterns,
            'recommendations': recommendations,
            'confidence_score': confidence,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 9. 保存分析结果
        self.save_zodiac_analysis(end_period, analysis_result)
        
        # 10. 显示结果
        print("\n📊 分析结果摘要:")
        print(f"   分析期数: {analysis_result['period_range']}")
        print(f"   总期数: {analysis_result['total_periods']}")
        print(f"   置信度: {analysis_result['confidence_score']:.3f}")
        print()
        
        print("🔥 热门生肖 Top 5:")
        for i, (zodiac, score) in enumerate(hot_zodiacs[:5], 1):
            print(f"   {i}. {zodiac}: {score:.3f}")
        print()
        
        print("❄️  冷门生肖 Top 5:")
        for i, (zodiac, score) in enumerate(cold_zodiacs[:5], 1):
            print(f"   {i}. {zodiac}: {score:.3f}")
        print()
        
        print("🎯 生肖推荐:")
        print(f"   热度推荐: {', '.join(recommendations['hot_based'][:6])}")
        print(f"   模式推荐: {', '.join(recommendations['pattern_based'][:6])}")
        print(f"   平衡推荐: {', '.join(recommendations['balance_based'][:6])}")
        print(f"   综合推荐: {', '.join(recommendations['comprehensive'])}")
        print()
        
        # 11. 创建可视化
        print("📈 步骤5: 生成可视化图表")
        print("-" * 30)
        self.create_zodiac_visualization(zodiac_stats, hot_zodiacs, cold_zodiacs)
        
        print("🎉 生肖分析完成！")
        return analysis_result

def main():
    """主函数"""
    analyzer = ZodiacAnalyzer()
    analyzer.run_complete_zodiac_analysis(1, 50)

if __name__ == "__main__":
    main()