#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
澳门六合彩分析系统主程序
功能：爬取数据、分析号码、生成推荐、可视化结果
"""

import argparse
import sys
import os
from datetime import datetime
import schedule
import time

from crawler import MacauLotteryCrawler
from analyzer import LotteryAnalyzer
from visualizer import LotteryVisualizer
from database import LotteryDatabase
from config import *

class MacauLotterySystem:
    def __init__(self):
        self.crawler = MacauLotteryCrawler()
        self.analyzer = LotteryAnalyzer()
        self.visualizer = LotteryVisualizer()
        self.db = LotteryDatabase()
    
    def crawl_data(self, start_period=1, end_period=289):
        """爬取数据"""
        print("=" * 50)
        print("开始爬取澳门六合彩数据")
        print("=" * 50)
        
        success_count, failed_periods = self.crawler.crawl_all_periods(start_period, end_period)
        
        if success_count > 0:
            # 保存为Excel文件
            excel_file = self.crawler.save_to_excel(start_period, end_period)
            print(f"数据已保存到Excel文件: {excel_file}")
        
        return success_count, failed_periods
    
    def analyze_data(self, start_period=1, end_period=289):
        """分析数据"""
        print("=" * 50)
        print("开始分析澳门六合彩数据")
        print("=" * 50)
        
        analysis_result = self.analyzer.run_complete_analysis(start_period, end_period)
        
        if analysis_result:
            print("\n分析结果摘要:")
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
    
    def visualize_results(self, analysis_result):
        """可视化结果"""
        if not analysis_result:
            print("没有分析结果可供可视化")
            return
        
        print("=" * 50)
        print("生成可视化图表")
        print("=" * 50)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 频率分析图
        freq_path = f"{REPORTS_DIR}/frequency_analysis_{timestamp}.png"
        self.visualizer.plot_frequency_analysis(
            analysis_result['frequency_stats'], freq_path
        )
        
        # 热门冷门号码图
        hot_cold_path = f"{REPORTS_DIR}/hot_cold_numbers_{timestamp}.png"
        self.visualizer.plot_hot_cold_numbers(
            analysis_result['hot_numbers'], 
            analysis_result['cold_numbers'], 
            hot_cold_path
        )
        
        # 模式分析图
        pattern_path = f"{REPORTS_DIR}/pattern_analysis_{timestamp}.png"
        self.visualizer.plot_pattern_analysis(
            analysis_result['patterns'], pattern_path
        )
        
        # 交互式仪表板
        dashboard_path = f"{REPORTS_DIR}/interactive_dashboard_{timestamp}.html"
        self.visualizer.create_interactive_dashboard(analysis_result, dashboard_path)
        
        # 综合分析报告
        report_path = f"{REPORTS_DIR}/comprehensive_report_{timestamp}.html"
        self.visualizer.generate_comprehensive_report(analysis_result, report_path)
        
        print(f"所有图表和报告已保存到 {REPORTS_DIR} 目录")
    
    def generate_test_data(self, start_period=1, end_period=289):
        """生成测试数据"""
        print("=" * 50)
        print("生成测试数据")
        print("=" * 50)
        
        success_count = 0
        for period in range(start_period, end_period + 1):
            data = self.crawler.generate_mock_data(period)
            if self.db.save_lottery_record(
                data['period'],
                data['draw_date'],
                data['numbers'],
                data['special_number']
            ):
                success_count += 1
        
        print(f"成功生成 {success_count} 期测试数据")
        return success_count
    
    def run_complete_analysis(self, start_period=1, end_period=289, use_test_data=False):
        """运行完整分析流程"""
        print("澳门六合彩分析系统启动")
        print(f"分析期数: {start_period} - {end_period}")
        print(f"使用测试数据: {use_test_data}")
        
        # 1. 爬取或生成数据
        if use_test_data:
            self.generate_test_data(start_period, end_period)
        else:
            success_count, failed_periods = self.crawl_data(start_period, end_period)
            if success_count == 0:
                print("爬取失败，使用测试数据")
                self.generate_test_data(start_period, end_period)
        
        # 2. 分析数据
        analysis_result = self.analyze_data(start_period, end_period)
        
        # 3. 可视化结果
        if analysis_result:
            self.visualize_results(analysis_result)
        
        print("=" * 50)
        print("分析完成！")
        print("=" * 50)
        
        return analysis_result
    
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
        success_count, failed_periods = self.crawler.crawl_all_periods(next_period, next_period)
        
        if success_count > 0:
            # 重新分析
            analysis_result = self.analyzer.run_complete_analysis()
            if analysis_result:
                self.visualizer.generate_comprehensive_report(analysis_result)
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
    parser = argparse.ArgumentParser(description='澳门六合彩分析系统')
    parser.add_argument('--start', type=int, default=1, help='开始期数 (默认: 1)')
    parser.add_argument('--end', type=int, default=289, help='结束期数 (默认: 289)')
    parser.add_argument('--test', action='store_true', help='使用测试数据')
    parser.add_argument('--crawl-only', action='store_true', help='仅爬取数据')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析数据')
    parser.add_argument('--auto-update', action='store_true', help='启动自动更新服务')
    
    args = parser.parse_args()
    
    system = MacauLotterySystem()
    
    if args.auto_update:
        system.start_auto_update_service()
    elif args.crawl_only:
        system.crawl_data(args.start, args.end)
    elif args.analyze_only:
        system.analyze_data(args.start, args.end)
    else:
        system.run_complete_analysis(args.start, args.end, args.test)

if __name__ == "__main__":
    main()