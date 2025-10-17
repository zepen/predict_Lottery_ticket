import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import REPORTS_DIR
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LotteryVisualizer:
    def __init__(self):
        self.reports_dir = REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def plot_frequency_analysis(self, frequency_stats, save_path=None):
        """绘制频率分析图"""
        numbers = list(frequency_stats.keys())
        frequencies = [stats['frequency'] for stats in frequency_stats.values()]
        hot_scores = [stats['hot_score'] for stats in frequency_stats.values()]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 频率柱状图
        bars1 = ax1.bar(numbers, frequencies, color='skyblue', alpha=0.7)
        ax1.set_title('号码出现频率统计', fontsize=16, fontweight='bold')
        ax1.set_xlabel('号码', fontsize=12)
        ax1.set_ylabel('出现次数', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, freq in zip(bars1, frequencies):
            if freq > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(freq), ha='center', va='bottom', fontsize=8)
        
        # 热度分数图
        bars2 = ax2.bar(numbers, hot_scores, color='coral', alpha=0.7)
        ax2.set_title('号码热度分数', fontsize=16, fontweight='bold')
        ax2.set_xlabel('号码', fontsize=12)
        ax2.set_ylabel('热度分数', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加热度阈值线
        ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='热门阈值')
        ax2.axhline(y=0.3, color='blue', linestyle='--', alpha=0.7, label='冷门阈值')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"频率分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_hot_cold_numbers(self, hot_numbers, cold_numbers, save_path=None):
        """绘制热门冷门号码图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 热门号码
        if hot_numbers:
            hot_nums, hot_scores = zip(*hot_numbers[:10])
            bars1 = ax1.bar(range(len(hot_nums)), hot_scores, color='red', alpha=0.7)
            ax1.set_title('热门号码 Top 10', fontsize=14, fontweight='bold')
            ax1.set_xlabel('排名', fontsize=12)
            ax1.set_ylabel('热度分数', fontsize=12)
            ax1.set_xticks(range(len(hot_nums)))
            ax1.set_xticklabels(hot_nums)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars1, hot_scores)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 冷门号码
        if cold_numbers:
            cold_nums, cold_scores = zip(*cold_numbers[:10])
            bars2 = ax2.bar(range(len(cold_nums)), cold_scores, color='blue', alpha=0.7)
            ax2.set_title('冷门号码 Top 10', fontsize=14, fontweight='bold')
            ax2.set_xlabel('排名', fontsize=12)
            ax2.set_ylabel('冷度分数', fontsize=12)
            ax2.set_xticks(range(len(cold_nums)))
            ax2.set_xticklabels(cold_nums)
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars2, cold_scores)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热门冷门号码图已保存到: {save_path}")
        
        plt.show()
    
    def plot_alternating_matrix(self, matrix, save_path=None):
        """绘制交替矩阵热力图"""
        plt.figure(figsize=(12, 10))
        
        # 创建热力图
        sns.heatmap(matrix, annot=False, cmap='YlOrRd', square=True, 
                   cbar_kws={'label': '共同出现次数'})
        
        plt.title('号码交替矩阵热力图', fontsize=16, fontweight='bold')
        plt.xlabel('号码', fontsize=12)
        plt.ylabel('号码', fontsize=12)
        
        # 设置坐标轴标签
        plt.xticks(range(0, 49, 5), range(1, 50, 5))
        plt.yticks(range(0, 49, 5), range(1, 50, 5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"交替矩阵图已保存到: {save_path}")
        
        plt.show()
    
    def plot_pattern_analysis(self, patterns, save_path=None):
        """绘制模式分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 连续号码对
        if 'consecutive_pairs' in patterns and patterns['consecutive_pairs']:
            pairs = list(patterns['consecutive_pairs'].items())[:10]
            pair_labels = [f"{p[0]}-{p[1]}" for p, _ in pairs]
            pair_counts = [count for _, count in pairs]
            
            axes[0, 0].bar(range(len(pair_labels)), pair_counts, color='green', alpha=0.7)
            axes[0, 0].set_title('连续号码对出现次数', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('号码对', fontsize=10)
            axes[0, 0].set_ylabel('出现次数', fontsize=10)
            axes[0, 0].set_xticks(range(len(pair_labels)))
            axes[0, 0].set_xticklabels(pair_labels, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 号码和分布
        if 'sum_ranges' in patterns and 'sum_distribution' in patterns['sum_ranges']:
            sum_dist = patterns['sum_ranges']['sum_distribution']
            sums = list(sum_dist.keys())
            counts = list(sum_dist.values())
            
            axes[0, 1].hist(sums, bins=20, weights=counts, color='orange', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('号码和分布', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('号码和', fontsize=10)
            axes[0, 1].set_ylabel('出现次数', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 奇偶比例
        if 'odd_even_ratio' in patterns and 'common_ratio' in patterns['odd_even_ratio']:
            ratios = patterns['odd_even_ratio']['common_ratio'][:8]
            ratio_labels = [f"{odd}奇{even}偶" for odd, even in ratios]
            ratio_counts = [1 for _ in ratios]  # 简化处理
            
            axes[1, 0].bar(range(len(ratio_labels)), ratio_counts, color='purple', alpha=0.7)
            axes[1, 0].set_title('奇偶比例分布', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('奇偶比例', fontsize=10)
            axes[1, 0].set_ylabel('出现次数', fontsize=10)
            axes[1, 0].set_xticks(range(len(ratio_labels)))
            axes[1, 0].set_xticklabels(ratio_labels, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 号码间隔分布
        if 'number_gaps' in patterns and 'gap_distribution' in patterns['number_gaps']:
            gap_dist = patterns['number_gaps']['gap_distribution']
            gaps = list(gap_dist.keys())
            counts = list(gap_dist.values())
            
            axes[1, 1].bar(gaps, counts, color='brown', alpha=0.7)
            axes[1, 1].set_title('号码间隔分布', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('间隔大小', fontsize=10)
            axes[1, 1].set_ylabel('出现次数', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模式分析图已保存到: {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, analysis_result, save_path=None):
        """创建交互式仪表板"""
        frequency_stats = analysis_result['frequency_stats']
        hot_numbers = analysis_result['hot_numbers']
        cold_numbers = analysis_result['cold_numbers']
        recommendations = analysis_result['recommendations']
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('号码频率统计', '热门冷门号码', '推荐号码', '热度分布'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 频率统计
        numbers = list(frequency_stats.keys())
        frequencies = [stats['frequency'] for stats in frequency_stats.values()]
        
        fig.add_trace(
            go.Bar(x=numbers, y=frequencies, name='出现次数', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 热门冷门号码
        if hot_numbers:
            hot_nums, hot_scores = zip(*hot_numbers[:10])
            fig.add_trace(
                go.Bar(x=list(hot_nums), y=list(hot_scores), name='热门号码', marker_color='red'),
                row=1, col=2
            )
        
        if cold_numbers:
            cold_nums, cold_scores = zip(*cold_numbers[:10])
            fig.add_trace(
                go.Bar(x=list(cold_nums), y=list(cold_scores), name='冷门号码', marker_color='blue'),
                row=1, col=2
            )
        
        # 推荐号码
        if recommendations['comprehensive']:
            rec_nums = recommendations['comprehensive']
            rec_scores = [frequency_stats[num]['hot_score'] for num in rec_nums if num in frequency_stats]
            
            fig.add_trace(
                go.Bar(x=rec_nums, y=rec_scores, name='推荐号码', marker_color='green'),
                row=2, col=1
            )
        
        # 热度分布
        hot_scores = [stats['hot_score'] for stats in frequency_stats.values()]
        fig.add_trace(
            go.Histogram(x=hot_scores, name='热度分布', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="澳门六合彩分析仪表板",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式仪表板已保存到: {save_path}")
        
        fig.show()
    
    def generate_comprehensive_report(self, analysis_result, save_path=None):
        """生成综合分析报告"""
        report_path = save_path or f"{self.reports_dir}/comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>澳门六合彩分析报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .hot {{ color: red; font-weight: bold; }}
                .cold {{ color: blue; font-weight: bold; }}
                .recommendation {{ color: green; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>澳门六合彩分析报告</h1>
                <p>分析期数: {analysis_result['period_range']}</p>
                <p>总期数: {analysis_result['total_periods']}</p>
                <p>分析时间: {analysis_result['analysis_date']}</p>
                <p>置信度: {analysis_result['confidence_score']:.3f}</p>
            </div>
            
            <div class="section">
                <h2>热门号码 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>号码</th><th>热度分数</th></tr>
        """
        
        for i, (num, score) in enumerate(analysis_result['hot_numbers'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='hot'>{num}</td><td>{score:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>冷门号码 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>号码</th><th>冷度分数</th></tr>
        """
        
        for i, (num, score) in enumerate(analysis_result['cold_numbers'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='cold'>{num}</td><td>{score:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>推荐号码</h2>
                <h3>综合推荐</h3>
                <p class="recommendation">
        """
        
        if analysis_result['recommendations']['comprehensive']:
            rec_nums = analysis_result['recommendations']['comprehensive']
            html_content += f"推荐号码: {', '.join(map(str, rec_nums))}"
        
        html_content += """
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"综合分析报告已保存到: {report_path}")
        return report_path