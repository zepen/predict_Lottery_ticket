#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
澳门六合彩专用分析系统主程序
专门针对六合彩的完整分析系统，包含生肖属性分析
"""

import argparse
import sys
import os
from datetime import datetime
import schedule
import time

from zodiac_analyzer import ZodiacAnalyzer
from analyzer import LotteryAnalyzer
from visualizer import LotteryVisualizer
from database import LotteryDatabase
from config import *

class MacauLotterySystem:
    def __init__(self):
        self.zodiac_analyzer = ZodiacAnalyzer()
        self.analyzer = LotteryAnalyzer()
        self.visualizer = LotteryVisualizer()
        self.db = LotteryDatabase()
    
    def crawl_data(self, start_period=1, end_period=289):
        """爬取六合彩数据"""
        print("=" * 50)
        print("开始爬取澳门六合彩数据")
        print("=" * 50)
        
        success_count = 0
        failed_periods = []
        
        for period in range(start_period, end_period + 1):
            print(f"正在爬取第{period}期数据...")
            
            # 使用生肖分析器的爬取功能
            data = self.zodiac_analyzer.crawl_lottery_data(period)
            if data:
                # 保存到主数据库
                if self.db.save_lottery_record(
                    data['period'],
                    data['draw_date'],
                    data['numbers'],
                    data['special_number']
                ):
                    # 保存到生肖数据库
                    self.zodiac_analyzer.save_lottery_record(data)
                    success_count += 1
                    print(f"第{period}期数据保存成功")
                else:
                    failed_periods.append(period)
            else:
                failed_periods.append(period)
            
            # 添加延迟避免被封
            time.sleep(1)
        
        print(f"爬取完成！成功: {success_count}, 失败: {len(failed_periods)}")
        if failed_periods:
            print(f"失败的期数: {failed_periods}")
        
        return success_count, failed_periods
    
    def analyze_data(self, start_period=1, end_period=289):
        """分析六合彩数据"""
        print("=" * 50)
        print("开始分析澳门六合彩数据")
        print("=" * 50)
        
        # 运行传统分析
        analysis_result = self.analyzer.run_complete_analysis(start_period, end_period)
        
        if analysis_result:
            print("\n传统分析结果摘要:")
            print(f"分析期数: {analysis_result['period_range']}")
            print(f"总期数: {analysis_result['total_periods']}")
            print(f"置信度: {analysis_result['confidence_score']:.3f}")
            
            print("\n热门号码 Top 5:")
            for i, (num, score) in enumerate(analysis_result['hot_numbers'][:5], 1):
                print(f"{i}. 号码 {num}: {score:.3f}")
            
            print("\n冷门号码 Top 5:")
            for i, (num, score) in enumerate(analysis_result['cold_numbers'][:5], 1):
                print(f"{i}. 号码 {num}: {score:.3f}")
            
            print("\n推荐号码:")
            if analysis_result['recommendations']['comprehensive']:
                rec_nums = analysis_result['recommendations']['comprehensive']
                print(f"综合推荐: {', '.join(map(str, rec_nums))}")
            
            return analysis_result
        
        return None
    
    def analyze_zodiac_data(self, start_period=1, end_period=289):
        """分析生肖数据"""
        print("=" * 50)
        print("开始分析六合彩生肖属性")
        print("=" * 50)
        
        # 运行生肖分析
        zodiac_result = self.zodiac_analyzer.run_complete_zodiac_analysis(start_period, end_period)
        
        if zodiac_result:
            print("\n生肖分析结果摘要:")
            print(f"分析期数: {zodiac_result['period_range']}")
            print(f"总期数: {zodiac_result['total_periods']}")
            print(f"置信度: {zodiac_result['confidence_score']:.3f}")
            
            print("\n热门生肖 Top 5:")
            for i, (zodiac, score) in enumerate(zodiac_result['hot_zodiacs'][:5], 1):
                print(f"{i}. {zodiac}: {score:.3f}")
            
            print("\n冷门生肖 Top 5:")
            for i, (zodiac, score) in enumerate(zodiac_result['cold_zodiacs'][:5], 1):
                print(f"{i}. {zodiac}: {score:.3f}")
            
            print("\n生肖推荐:")
            recommendations = zodiac_result['recommendations']
            print(f"热度推荐: {', '.join(recommendations['hot_based'][:6])}")
            print(f"模式推荐: {', '.join(recommendations['pattern_based'][:6])}")
            print(f"平衡推荐: {', '.join(recommendations['balance_based'][:6])}")
            print(f"综合推荐: {', '.join(recommendations['comprehensive'])}")
            
            return zodiac_result
        
        return None
    
    def generate_test_data(self, start_period=1, end_period=289):
        """生成测试数据"""
        print("=" * 50)
        print("生成六合彩测试数据")
        print("=" * 50)
        
        success_count = 0
        for period in range(start_period, end_period + 1):
            # 生成传统数据
            data = self.zodiac_analyzer.generate_mock_data(period)
            
            # 保存到主数据库
            if self.db.save_lottery_record(
                data['period'],
                data['draw_date'],
                data['numbers'],
                data['special_number']
            ):
                # 保存到生肖数据库
                self.zodiac_analyzer.save_lottery_record(data)
                success_count += 1
        
        print(f"成功生成 {success_count} 期测试数据")
        return success_count
    
    def run_complete_analysis(self, start_period=1, end_period=289, use_test_data=False):
        """运行完整分析流程"""
        print("🐉 澳门六合彩专用分析系统")
        print("=" * 60)
        print(f"分析期数: {start_period} - {end_period}")
        print(f"使用测试数据: {use_test_data}")
        print()
        
        # 1. 爬取或生成数据
        if use_test_data:
            self.generate_test_data(start_period, end_period)
        else:
            success_count, failed_periods = self.crawl_data(start_period, end_period)
            if success_count == 0:
                print("爬取失败，使用测试数据")
                self.generate_test_data(start_period, end_period)
        
        # 2. 传统数据分析
        print("\n" + "=" * 60)
        print("📊 传统数据分析")
        print("=" * 60)
        analysis_result = self.analyze_data(start_period, end_period)
        
        # 3. 生肖属性分析
        print("\n" + "=" * 60)
        print("🐉 生肖属性分析")
        print("=" * 60)
        zodiac_result = self.analyze_zodiac_data(start_period, end_period)
        
        # 4. 综合报告
        print("\n" + "=" * 60)
        print("📋 综合分析报告")
        print("=" * 60)
        
        if analysis_result and zodiac_result:
            print("✅ 传统分析完成")
            print("✅ 生肖分析完成")
            
            # 生成综合推荐
            print("\n🎯 综合推荐:")
            print(f"传统推荐号码: {', '.join(map(str, analysis_result['recommendations']['comprehensive']))}")
            print(f"生肖推荐: {', '.join(zodiac_result['recommendations']['comprehensive'])}")
            
            # 生成综合报告
            self.generate_comprehensive_report(analysis_result, zodiac_result)
        
        print("\n" + "=" * 60)
        print("🎉 六合彩分析完成！")
        print("=" * 60)
        
        return analysis_result, zodiac_result
    
    def generate_comprehensive_report(self, analysis_result, zodiac_result):
        """生成综合分析报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{REPORTS_DIR}/comprehensive_macau_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>澳门六合彩综合分析报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .hot {{ color: red; font-weight: bold; }}
                .cold {{ color: blue; font-weight: bold; }}
                .recommendation {{ color: green; font-weight: bold; }}
                .zodiac {{ color: purple; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🐉 澳门六合彩综合分析报告</h1>
                <p>分析期数: {analysis_result['period_range']}</p>
                <p>总期数: {analysis_result['total_periods']}</p>
                <p>分析时间: {analysis_result['analysis_date']}</p>
                <p>传统分析置信度: {analysis_result['confidence_score']:.3f}</p>
                <p>生肖分析置信度: {zodiac_result['confidence_score']:.3f}</p>
            </div>
            
            <div class="section">
                <h2>🔥 热门号码 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>号码</th><th>热度分数</th></tr>
        """
        
        for i, (num, score) in enumerate(analysis_result['hot_numbers'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='hot'>{num}</td><td>{score:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>❄️ 冷门号码 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>号码</th><th>冷度分数</th></tr>
        """
        
        for i, (num, score) in enumerate(analysis_result['cold_numbers'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='cold'>{num}</td><td>{score:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>🐉 热门生肖 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>生肖</th><th>热度分数</th></tr>
        """
        
        for i, (zodiac, score) in enumerate(zodiac_result['hot_zodiacs'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='zodiac'>{zodiac}</td><td>{score:.3f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>❄️ 冷门生肖 Top 10</h2>
                <table>
                    <tr><th>排名</th><th>生肖</th><th>冷度分数</th></tr>
        """
        
        for i, (zodiac, score) in enumerate(zodiac_result['cold_zodiacs'][:10], 1):
            html_content += f"<tr><td>{i}</td><td class='zodiac'>{zodiac}</td><td>{score:.3f}</td></tr>"
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 综合推荐</h2>
                <h3>传统推荐</h3>
                <p class="recommendation">
                    推荐号码: {', '.join(map(str, analysis_result['recommendations']['comprehensive']))}
                </p>
                
                <h3>生肖推荐</h3>
                <p class="zodiac">
                    推荐生肖: {', '.join(zodiac_result['recommendations']['comprehensive'])}
                </p>
                
                <h3>综合建议</h3>
                <p>结合传统分析和生肖属性分析，建议重点关注以上推荐的号码和生肖组合。</p>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 综合分析报告已保存: {report_path}")
        return report_path
    
    def auto_update(self):
        """自动更新数据"""
        print(f"自动更新数据 - {datetime.now()}")
        
        # 获取最新期数
        df = self.db.get_lottery_records()
        if not df.empty:
            last_period = df['period'].max()
            next_period = last_period + 1
        else:
            next_period = 1
        
        # 爬取新数据
        success_count, failed_periods = self.crawl_data(next_period, next_period)
        
        if success_count > 0:
            # 重新分析
            analysis_result = self.analyzer.run_complete_analysis()
            zodiac_result = self.zodiac_analyzer.run_complete_zodiac_analysis(next_period, next_period)
            
            if analysis_result and zodiac_result:
                self.generate_comprehensive_report(analysis_result, zodiac_result)
                print("自动更新完成")
        else:
            print("自动更新失败")
    
    def start_auto_update_service(self):
        """启动自动更新服务"""
        print("启动自动更新服务...")
        print("每天 21:00 自动更新数据")
        
        # 每天21:00执行自动更新
        schedule.every().day.at("21:00").do(self.auto_update)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次

def main():
    parser = argparse.ArgumentParser(description='澳门六合彩专用分析系统')
    parser.add_argument('--start', type=int, default=1, help='开始期数 (默认: 1)')
    parser.add_argument('--end', type=int, default=289, help='结束期数 (默认: 289)')
    parser.add_argument('--test', action='store_true', help='使用测试数据')
    parser.add_argument('--crawl-only', action='store_true', help='仅爬取数据')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析数据')
    parser.add_argument('--zodiac-only', action='store_true', help='仅分析生肖数据')
    parser.add_argument('--auto-update', action='store_true', help='启动自动更新服务')
    
    args = parser.parse_args()
    
    system = MacauLotterySystem()
    
    if args.auto_update:
        system.start_auto_update_service()
    elif args.crawl_only:
        system.crawl_data(args.start, args.end)
    elif args.analyze_only:
        system.analyze_data(args.start, args.end)
    elif args.zodiac_only:
        system.analyze_zodiac_data(args.start, args.end)
    else:
        system.run_complete_analysis(args.start, args.end, args.test)

if __name__ == "__main__":
    main()