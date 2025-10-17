import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from database import LotteryDatabase
from config import NUMBER_RANGE, HOT_THRESHOLD, COLD_THRESHOLD, ANALYSIS_PERIODS
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class LotteryAnalyzer:
    def __init__(self):
        self.db = LotteryDatabase()
        self.number_range = NUMBER_RANGE
        self.hot_threshold = HOT_THRESHOLD
        self.cold_threshold = COLD_THRESHOLD
    
    def get_lottery_data(self, start_period=1, end_period=289):
        """获取开奖数据"""
        df = self.db.get_lottery_records(start_period, end_period)
        return df
    
    def parse_numbers(self, numbers_str):
        """解析号码字符串为列表"""
        if pd.isna(numbers_str):
            return []
        return [int(x) for x in numbers_str.split(',')]
    
    def calculate_frequency_analysis(self, df):
        """计算号码频率分析"""
        all_numbers = []
        number_periods = defaultdict(list)  # 记录每个号码出现的期数
        
        for _, row in df.iterrows():
            numbers = self.parse_numbers(row['numbers'])
            period = row['period']
            all_numbers.extend(numbers)
            
            for num in numbers:
                number_periods[num].append(period)
        
        # 计算频率统计
        frequency_stats = {}
        total_periods = len(df)
        
        for num in self.number_range:
            count = all_numbers.count(num)
            frequency = count / total_periods if total_periods > 0 else 0
            last_appeared = max(number_periods[num]) if number_periods[num] else 0
            
            # 计算热度分数（基于频率和最近出现情况）
            recent_periods = [p for p in number_periods[num] if p > total_periods - 50]
            recent_frequency = len(recent_periods) / 50 if total_periods >= 50 else frequency
            
            hot_score = (frequency * 0.7 + recent_frequency * 0.3)
            cold_score = 1 - hot_score
            
            frequency_stats[num] = {
                'frequency': count,
                'frequency_rate': frequency,
                'last_appeared_period': last_appeared,
                'hot_score': hot_score,
                'cold_score': cold_score,
                'appearances': number_periods[num]
            }
            
            # 更新数据库
            self.db.update_number_frequency(
                num, count, last_appeared, hot_score, cold_score
            )
        
        return frequency_stats
    
    def identify_hot_cold_numbers(self, frequency_stats):
        """识别热门和冷门号码"""
        hot_numbers = []
        cold_numbers = []
        
        for num, stats in frequency_stats.items():
            if stats['hot_score'] >= self.hot_threshold:
                hot_numbers.append((num, stats['hot_score']))
            elif stats['cold_score'] >= self.cold_threshold:
                cold_numbers.append((num, stats['cold_score']))
        
        # 按分数排序
        hot_numbers.sort(key=lambda x: x[1], reverse=True)
        cold_numbers.sort(key=lambda x: x[1], reverse=True)
        
        return hot_numbers, cold_numbers
    
    def analyze_number_patterns(self, df):
        """分析号码模式"""
        patterns = {
            'consecutive_pairs': self.find_consecutive_pairs(df),
            'sum_ranges': self.analyze_sum_ranges(df),
            'odd_even_ratio': self.analyze_odd_even_ratio(df),
            'number_gaps': self.analyze_number_gaps(df)
        }
        return patterns
    
    def find_consecutive_pairs(self, df):
        """查找连续号码对"""
        consecutive_pairs = Counter()
        
        for _, row in df.iterrows():
            numbers = sorted(self.parse_numbers(row['numbers']))
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    pair = (numbers[i], numbers[i+1])
                    consecutive_pairs[pair] += 1
        
        return dict(consecutive_pairs.most_common(10))
    
    def analyze_sum_ranges(self, df):
        """分析号码和的范围"""
        sums = []
        for _, row in df.iterrows():
            numbers = self.parse_numbers(row['numbers'])
            if numbers:
                sums.append(sum(numbers))
        
        if sums:
            return {
                'min_sum': min(sums),
                'max_sum': max(sums),
                'avg_sum': np.mean(sums),
                'sum_distribution': Counter(sums)
            }
        return {}
    
    def analyze_odd_even_ratio(self, df):
        """分析奇偶比例"""
        odd_even_ratios = []
        
        for _, row in df.iterrows():
            numbers = self.parse_numbers(row['numbers'])
            if numbers:
                odd_count = sum(1 for num in numbers if num % 2 == 1)
                even_count = len(numbers) - odd_count
                odd_even_ratios.append((odd_count, even_count))
        
        if odd_even_ratios:
            avg_odd = np.mean([ratio[0] for ratio in odd_even_ratios])
            avg_even = np.mean([ratio[1] for ratio in odd_even_ratios])
            return {
                'avg_odd_count': avg_odd,
                'avg_even_count': avg_even,
                'common_ratio': Counter(odd_even_ratios).most_common(5)
            }
        return {}
    
    def analyze_number_gaps(self, df):
        """分析号码间隔"""
        gaps = []
        
        for _, row in df.iterrows():
            numbers = sorted(self.parse_numbers(row['numbers']))
            if len(numbers) > 1:
                for i in range(len(numbers) - 1):
                    gap = numbers[i+1] - numbers[i]
                    gaps.append(gap)
        
        if gaps:
            return {
                'avg_gap': np.mean(gaps),
                'min_gap': min(gaps),
                'max_gap': max(gaps),
                'gap_distribution': Counter(gaps)
            }
        return {}
    
    def create_alternating_matrix(self, df):
        """创建交替矩阵分析"""
        matrix = np.zeros((49, 49))
        
        for _, row in df.iterrows():
            numbers = self.parse_numbers(row['numbers'])
            if len(numbers) >= 2:
                for i in range(len(numbers)):
                    for j in range(i+1, len(numbers)):
                        num1, num2 = numbers[i] - 1, numbers[j] - 1  # 转换为0索引
                        matrix[num1][num2] += 1
                        matrix[num2][num1] += 1  # 对称矩阵
        
        return matrix
    
    def generate_recommendations(self, frequency_stats, patterns, matrix):
        """生成推荐号码"""
        recommendations = {
            'hot_based': [],
            'pattern_based': [],
            'matrix_based': [],
            'comprehensive': []
        }
        
        # 基于热度的推荐
        hot_numbers = [num for num, score in sorted(
            [(num, stats['hot_score']) for num, stats in frequency_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:15]]
        
        recommendations['hot_based'] = hot_numbers[:6]
        
        # 基于模式的推荐
        if 'consecutive_pairs' in patterns:
            consecutive_nums = set()
            for pair, count in list(patterns['consecutive_pairs'].items())[:5]:
                consecutive_nums.update(pair)
            recommendations['pattern_based'] = list(consecutive_nums)[:6]
        
        # 基于矩阵的推荐
        if matrix is not None:
            # 找出最常一起出现的号码对
            matrix_sum = np.sum(matrix, axis=1)
            top_numbers = np.argsort(matrix_sum)[-15:]
            recommendations['matrix_based'] = [num + 1 for num in top_numbers[:6]]
        
        # 综合推荐
        all_candidates = set()
        for rec_type in ['hot_based', 'pattern_based', 'matrix_based']:
            all_candidates.update(recommendations[rec_type])
        
        # 计算综合分数
        candidate_scores = {}
        for num in all_candidates:
            if num in frequency_stats:
                score = frequency_stats[num]['hot_score']
                if num in recommendations['hot_based']:
                    score += 0.3
                if num in recommendations['pattern_based']:
                    score += 0.2
                if num in recommendations['matrix_based']:
                    score += 0.1
                candidate_scores[num] = score
        
        # 选择分数最高的6个号码
        comprehensive = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        recommendations['comprehensive'] = [num for num, score in comprehensive]
        
        return recommendations
    
    def calculate_confidence_score(self, recommendations, frequency_stats):
        """计算推荐置信度"""
        if not recommendations['comprehensive']:
            return 0.0
        
        scores = []
        for num in recommendations['comprehensive']:
            if num in frequency_stats:
                scores.append(frequency_stats[num]['hot_score'])
        
        if scores:
            avg_score = np.mean(scores)
            # 归一化到0-1范围
            confidence = min(avg_score * 2, 1.0)
            return round(confidence, 3)
        
        return 0.0
    
    def run_complete_analysis(self, start_period=1, end_period=289):
        """运行完整分析"""
        print(f"开始分析第{start_period}期到第{end_period}期的数据...")
        
        # 获取数据
        df = self.get_lottery_data(start_period, end_period)
        if df.empty:
            print("没有找到数据，请先运行爬虫")
            return None
        
        print(f"共分析{len(df)}期数据")
        
        # 频率分析
        print("进行频率分析...")
        frequency_stats = self.calculate_frequency_analysis(df)
        
        # 热门冷门号码
        print("识别热门冷门号码...")
        hot_numbers, cold_numbers = self.identify_hot_cold_numbers(frequency_stats)
        
        # 模式分析
        print("分析号码模式...")
        patterns = self.analyze_number_patterns(df)
        
        # 交替矩阵
        print("创建交替矩阵...")
        matrix = self.create_alternating_matrix(df)
        
        # 生成推荐
        print("生成推荐号码...")
        recommendations = self.generate_recommendations(frequency_stats, patterns, matrix)
        
        # 计算置信度
        confidence = self.calculate_confidence_score(recommendations, frequency_stats)
        
        # 保存分析结果
        analysis_result = {
            'period_range': f"{start_period}-{end_period}",
            'total_periods': len(df),
            'hot_numbers': hot_numbers[:10],
            'cold_numbers': cold_numbers[:10],
            'recommendations': recommendations,
            'confidence_score': confidence,
            'frequency_stats': frequency_stats,
            'patterns': patterns,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到数据库
        self.db.save_analysis_result(
            end_period,
            'comprehensive',
            ','.join(map(str, [num for num, _ in hot_numbers[:6]])),
            ','.join(map(str, [num for num, _ in cold_numbers[:6]])),
            ','.join(map(str, recommendations['comprehensive'])),
            confidence
        )
        
        return analysis_result