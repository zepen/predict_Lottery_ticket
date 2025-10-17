#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球与六合彩对比分析系统
分析两种彩票的不同程度和特点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LotteryComparison:
    def __init__(self):
        self.ssq_info = {
            'name': '双色球',
            'region': '中国大陆',
            'red_balls': list(range(1, 34)),  # 红球1-33
            'blue_balls': list(range(1, 17)),  # 蓝球1-16
            'red_count': 6,  # 选6个红球
            'blue_count': 1,  # 选1个蓝球
            'total_combinations': self.calculate_ssq_combinations(),
            'draw_frequency': '每周二、四、日',
            'prize_levels': 6
        }
        
        self.macau_info = {
            'name': '澳门六合彩',
            'region': '澳门',
            'numbers': list(range(1, 50)),  # 1-49
            'select_count': 6,  # 选6个号码
            'total_combinations': self.calculate_macau_combinations(),
            'draw_frequency': '每周二、四、六',
            'prize_levels': 7
        }
    
    def calculate_ssq_combinations(self):
        """计算双色球总组合数"""
        # C(33,6) * C(16,1) = 17,721,088
        from math import comb
        red_combinations = comb(33, 6)
        blue_combinations = comb(16, 1)
        return red_combinations * blue_combinations
    
    def calculate_macau_combinations(self):
        """计算澳门六合彩总组合数"""
        # C(49,6) = 13,983,816
        from math import comb
        return comb(49, 6)
    
    def analyze_probability_differences(self):
        """分析概率差异"""
        print("🎲 双色球与澳门六合彩概率对比分析")
        print("=" * 60)
        
        # 一等奖概率
        ssq_first_prize = 1 / self.ssq_info['total_combinations']
        macau_first_prize = 1 / self.macau_info['total_combinations']
        
        print(f"📊 一等奖中奖概率对比:")
        print(f"   双色球: 1 / {self.ssq_info['total_combinations']:,} = {ssq_first_prize:.2e}")
        print(f"   澳门六合彩: 1 / {self.macau_info['total_combinations']:,} = {macau_first_prize:.2e}")
        print(f"   概率比: {ssq_first_prize/macau_first_prize:.2f}:1 (双色球更难)")
        print()
        
        # 号码选择难度
        print(f"🎯 号码选择难度对比:")
        print(f"   双色球: 从{len(self.ssq_info['red_balls'])}个红球选{self.ssq_info['red_count']}个 + 从{len(self.ssq_info['blue_balls'])}个蓝球选{self.ssq_info['blue_count']}个")
        print(f"   澳门六合彩: 从{len(self.macau_info['numbers'])}个号码选{self.macau_info['select_count']}个")
        print(f"   选择复杂度: 双色球 > 澳门六合彩")
        print()
        
        return {
            'ssq_probability': ssq_first_prize,
            'macau_probability': macau_first_prize,
            'probability_ratio': ssq_first_prize / macau_first_prize
        }
    
    def analyze_number_distribution(self):
        """分析号码分布特点"""
        print("📈 号码分布特点分析")
        print("=" * 60)
        
        # 双色球分布特点
        print("🔴 双色球分布特点:")
        print(f"   红球范围: 1-33 (共33个)")
        print(f"   蓝球范围: 1-16 (共16个)")
        print(f"   号码分布: 红球和蓝球分别独立分布")
        print(f"   选择策略: 需要同时考虑红球和蓝球的组合")
        print()
        
        # 澳门六合彩分布特点
        print("🟡 澳门六合彩分布特点:")
        print(f"   号码范围: 1-49 (共49个)")
        print(f"   号码分布: 所有号码在同一池中")
        print(f"   选择策略: 只需考虑6个号码的组合")
        print()
        
        # 号码密度分析
        ssq_density = self.ssq_info['red_count'] / len(self.ssq_info['red_balls'])
        macau_density = self.macau_info['select_count'] / len(self.macau_info['numbers'])
        
        print("📊 号码密度对比:")
        print(f"   双色球红球密度: {ssq_density:.2%}")
        print(f"   澳门六合彩密度: {macau_density:.2%}")
        print(f"   密度比: {ssq_density/macau_density:.2f}:1")
        print()
    
    def analyze_prize_structure(self):
        """分析奖金结构"""
        print("💰 奖金结构对比")
        print("=" * 60)
        
        # 双色球奖金结构（示例）
        ssq_prizes = {
            '一等奖': {'match': '6红+1蓝', 'probability': 1/self.ssq_info['total_combinations']},
            '二等奖': {'match': '6红+0蓝', 'probability': 15/self.ssq_info['total_combinations']},
            '三等奖': {'match': '5红+1蓝', 'probability': 162/self.ssq_info['total_combinations']},
            '四等奖': {'match': '5红+0蓝', 'probability': 2430/self.ssq_info['total_combinations']},
            '五等奖': {'match': '4红+1蓝', 'probability': 52650/self.ssq_info['total_combinations']},
            '六等奖': {'match': '4红+0蓝', 'probability': 789750/self.ssq_info['total_combinations']}
        }
        
        # 澳门六合彩奖金结构（示例）
        macau_prizes = {
            '头奖': {'match': '6个号码全中', 'probability': 1/self.macau_info['total_combinations']},
            '二奖': {'match': '5个号码+特别号码', 'probability': 6/self.macau_info['total_combinations']},
            '三奖': {'match': '5个号码', 'probability': 258/self.macau_info['total_combinations']},
            '四奖': {'match': '4个号码+特别号码', 'probability': 1290/self.macau_info['total_combinations']},
            '五奖': {'match': '4个号码', 'probability': 13545/self.macau_info['total_combinations']},
            '六奖': {'match': '3个号码+特别号码', 'probability': 17220/self.macau_info['total_combinations']},
            '七奖': {'match': '3个号码', 'probability': 180810/self.macau_info['total_combinations']}
        }
        
        print("🔴 双色球奖金等级:")
        for level, info in ssq_prizes.items():
            print(f"   {level}: {info['match']} (概率: {info['probability']:.2e})")
        print()
        
        print("🟡 澳门六合彩奖金等级:")
        for level, info in macau_prizes.items():
            print(f"   {level}: {info['match']} (概率: {info['probability']:.2e})")
        print()
    
    def analyze_analysis_difficulty(self):
        """分析分析难度"""
        print("🔍 数据分析难度对比")
        print("=" * 60)
        
        print("📊 双色球分析特点:")
        print("   ✅ 红球和蓝球分别分析，相对简单")
        print("   ✅ 红球范围较小(1-33)，分析维度较少")
        print("   ✅ 历史数据丰富，分析模型成熟")
        print("   ❌ 需要考虑红蓝球组合，增加复杂度")
        print("   ❌ 两个独立号码池，关联性分析困难")
        print()
        
        print("📊 澳门六合彩分析特点:")
        print("   ✅ 单一号码池，分析逻辑统一")
        print("   ✅ 号码范围适中(1-49)，分析维度合理")
        print("   ✅ 号码间关联性强，模式分析丰富")
        print("   ❌ 号码范围较大，分析复杂度较高")
        print("   ❌ 需要更复杂的统计模型")
        print()
        
        # 分析复杂度评分
        ssq_complexity = 7  # 1-10分，10分最难
        macau_complexity = 6
        
        print("🎯 分析复杂度评分 (1-10分，10分最难):")
        print(f"   双色球: {ssq_complexity}/10")
        print(f"   澳门六合彩: {macau_complexity}/10")
        print(f"   双色球分析难度更高")
        print()
    
    def create_comparison_chart(self):
        """创建对比图表"""
        print("📈 生成对比分析图表")
        print("=" * 60)
        
        # 创建对比数据
        categories = ['中奖概率', '号码范围', '选择难度', '分析复杂度', '奖金等级']
        ssq_scores = [2, 8, 7, 7, 6]  # 1-10分，10分最难/最多
        macau_scores = [3, 6, 6, 6, 7]
        
        # 创建雷达图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw=dict(projection='polar'))
        
        # 双色球雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        ssq_scores += ssq_scores[:1]
        
        ax1.plot(angles, ssq_scores, 'o-', linewidth=2, label='双色球', color='red')
        ax1.fill(angles, ssq_scores, alpha=0.25, color='red')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 10)
        ax1.set_title('双色球特征分析', size=14, fontweight='bold')
        ax1.grid(True)
        
        # 澳门六合彩雷达图
        macau_scores += macau_scores[:1]
        ax2.plot(angles, macau_scores, 'o-', linewidth=2, label='澳门六合彩', color='orange')
        ax2.fill(angles, macau_scores, alpha=0.25, color='orange')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 10)
        ax2.set_title('澳门六合彩特征分析', size=14, fontweight='bold')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('lottery_comparison_radar.png', dpi=300, bbox_inches='tight')
        print("✅ 雷达图已保存: lottery_comparison_radar.png")
        
        # 创建概率对比柱状图
        plt.figure(figsize=(12, 8))
        
        lottery_names = ['双色球', '澳门六合彩']
        probabilities = [1/self.ssq_info['total_combinations'], 1/self.macau_info['total_combinations']]
        combinations = [self.ssq_info['total_combinations'], self.macau_info['total_combinations']]
        
        # 子图1: 中奖概率对比
        plt.subplot(2, 2, 1)
        bars = plt.bar(lottery_names, [p*1e8 for p in probabilities], color=['red', 'orange'], alpha=0.7)
        plt.title('一等奖中奖概率对比 (×10⁻⁸)', fontweight='bold')
        plt.ylabel('概率 (×10⁻⁸)')
        
        # 添加数值标签
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{prob*1e8:.2f}', ha='center', va='bottom')
        
        # 子图2: 总组合数对比
        plt.subplot(2, 2, 2)
        bars = plt.bar(lottery_names, [c/1e6 for c in combinations], color=['red', 'orange'], alpha=0.7)
        plt.title('总组合数对比 (×10⁶)', fontweight='bold')
        plt.ylabel('组合数 (×10⁶)')
        
        for bar, comb in zip(bars, combinations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{comb/1e6:.1f}', ha='center', va='bottom')
        
        # 子图3: 号码范围对比
        plt.subplot(2, 2, 3)
        red_balls = len(self.ssq_info['red_balls'])
        blue_balls = len(self.ssq_info['blue_balls'])
        macau_numbers = len(self.macau_info['numbers'])
        
        x = ['双色球红球', '双色球蓝球', '澳门六合彩']
        y = [red_balls, blue_balls, macau_numbers]
        colors = ['red', 'blue', 'orange']
        
        bars = plt.bar(x, y, color=colors, alpha=0.7)
        plt.title('号码范围对比', fontweight='bold')
        plt.ylabel('号码数量')
        
        for bar, num in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(num), ha='center', va='bottom')
        
        # 子图4: 分析难度评分
        plt.subplot(2, 2, 4)
        difficulty_scores = [7, 6]  # 双色球更难
        bars = plt.bar(lottery_names, difficulty_scores, color=['red', 'orange'], alpha=0.7)
        plt.title('分析难度评分 (1-10分)', fontweight='bold')
        plt.ylabel('难度分数')
        plt.ylim(0, 10)
        
        for bar, score in zip(bars, difficulty_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(score), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('lottery_comparison_charts.png', dpi=300, bbox_inches='tight')
        print("✅ 对比图表已保存: lottery_comparison_charts.png")
        plt.show()
    
    def generate_detailed_comparison_report(self):
        """生成详细对比报告"""
        print("📋 生成详细对比分析报告")
        print("=" * 60)
        
        report = {
            'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'lottery_comparison': {
                '双色球': {
                    '基本信息': {
                        '地区': '中国大陆',
                        '开奖频率': '每周二、四、日',
                        '号码范围': '红球1-33，蓝球1-16',
                        '选择数量': '6个红球+1个蓝球',
                        '总组合数': f"{self.ssq_info['total_combinations']:,}",
                        '一等奖概率': f"{1/self.ssq_info['total_combinations']:.2e}"
                    },
                    '分析特点': {
                        '优势': [
                            '红球和蓝球分别分析，逻辑清晰',
                            '红球范围较小，分析相对简单',
                            '历史数据丰富，分析模型成熟',
                            '奖金结构合理，中奖层次多'
                        ],
                        '劣势': [
                            '需要考虑红蓝球组合，增加复杂度',
                            '两个独立号码池，关联性分析困难',
                            '中奖概率较低，一等奖难度大',
                            '分析维度较多，需要更复杂的模型'
                        ]
                    }
                },
                '澳门六合彩': {
                    '基本信息': {
                        '地区': '澳门',
                        '开奖频率': '每周二、四、六',
                        '号码范围': '1-49',
                        '选择数量': '6个号码',
                        '总组合数': f"{self.macau_info['total_combinations']:,}",
                        '一等奖概率': f"{1/self.macau_info['total_combinations']:.2e}"
                    },
                    '分析特点': {
                        '优势': [
                            '单一号码池，分析逻辑统一',
                            '号码间关联性强，模式分析丰富',
                            '分析维度适中，模型相对简单',
                            '中奖概率相对较高'
                        ],
                        '劣势': [
                            '号码范围较大，分析复杂度较高',
                            '需要更复杂的统计模型',
                            '历史数据相对较少',
                            '奖金结构相对简单'
                        ]
                    }
                }
            },
            '对比结论': {
                '中奖概率': '澳门六合彩 > 双色球',
                '分析难度': '双色球 > 澳门六合彩',
                '号码复杂度': '双色球 > 澳门六合彩',
                '模式丰富度': '澳门六合彩 > 双色球',
                '推荐分析': '澳门六合彩更适合进行深度分析'
            }
        }
        
        # 保存报告
        with open('lottery_comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("✅ 详细对比报告已保存: lottery_comparison_report.json")
        
        return report
    
    def run_complete_comparison(self):
        """运行完整对比分析"""
        print("🎰 双色球与澳门六合彩深度对比分析")
        print("=" * 80)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 概率分析
        prob_results = self.analyze_probability_differences()
        
        # 2. 号码分布分析
        self.analyze_number_distribution()
        
        # 3. 奖金结构分析
        self.analyze_prize_structure()
        
        # 4. 分析难度分析
        self.analyze_analysis_difficulty()
        
        # 5. 创建对比图表
        self.create_comparison_chart()
        
        # 6. 生成详细报告
        report = self.generate_detailed_comparison_report()
        
        print("\n" + "=" * 80)
        print("🎯 对比分析总结")
        print("=" * 80)
        print("📊 关键发现:")
        print(f"   1. 中奖概率: 澳门六合彩比双色球高 {prob_results['probability_ratio']:.1f} 倍")
        print("   2. 分析难度: 双色球需要同时分析红球和蓝球，复杂度更高")
        print("   3. 号码关联: 澳门六合彩号码间关联性更强，模式分析更丰富")
        print("   4. 分析建议: 澳门六合彩更适合进行深度数据分析和预测")
        print()
        print("💡 推荐策略:")
        print("   - 如果追求更高的中奖概率: 选择澳门六合彩")
        print("   - 如果喜欢复杂的分析挑战: 选择双色球")
        print("   - 如果进行数据分析研究: 推荐澳门六合彩")
        print("   - 如果追求奖金最大化: 需要根据具体奖金设置判断")
        print()
        print("🎉 对比分析完成！")

def main():
    """主函数"""
    comparison = LotteryComparison()
    comparison.run_complete_comparison()

if __name__ == "__main__":
    main()