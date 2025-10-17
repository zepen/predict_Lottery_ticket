#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
澳门六合彩分析系统测试脚本
"""

import sys
import os
from datetime import datetime
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MacauLotterySystem
from database import LotteryDatabase
from analyzer import LotteryAnalyzer
from visualizer import LotteryVisualizer

def test_database():
    """测试数据库功能"""
    print("测试数据库功能...")
    try:
        db = LotteryDatabase()
        
        # 测试保存记录
        test_data = {
            'period': 999,
            'draw_date': '2025-01-01',
            'numbers': '1,2,3,4,5,6',
            'special_number': 7
        }
        
        result = db.save_lottery_record(
            test_data['period'],
            test_data['draw_date'],
            test_data['numbers'],
            test_data['special_number']
        )
        
        if result:
            print("✓ 数据库保存功能正常")
        else:
            print("✗ 数据库保存功能失败")
            return False
        
        # 测试读取记录
        df = db.get_lottery_records(999, 999)
        if not df.empty and df.iloc[0]['period'] == 999:
            print("✓ 数据库读取功能正常")
        else:
            print("✗ 数据库读取功能失败")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        traceback.print_exc()
        return False

def test_crawler():
    """测试爬虫功能"""
    print("测试爬虫功能...")
    try:
        from crawler import MacauLotteryCrawler
        crawler = MacauLotteryCrawler()
        
        # 测试生成模拟数据
        test_data = crawler.generate_mock_data(1)
        if test_data and 'period' in test_data and 'numbers' in test_data:
            print("✓ 模拟数据生成功能正常")
        else:
            print("✗ 模拟数据生成功能失败")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 爬虫测试失败: {e}")
        traceback.print_exc()
        return False

def test_analyzer():
    """测试分析器功能"""
    print("测试分析器功能...")
    try:
        analyzer = LotteryAnalyzer()
        
        # 创建测试数据
        import pandas as pd
        test_data = []
        for i in range(1, 11):  # 10期测试数据
            test_data.append({
                'period': i,
                'draw_date': f'2025-01-{i:02d}',
                'numbers': f'{i},{i+1},{i+2},{i+3},{i+4},{i+5}',
                'special_number': i + 10
            })
        
        df = pd.DataFrame(test_data)
        
        # 测试频率分析
        frequency_stats = analyzer.calculate_frequency_analysis(df)
        if frequency_stats and len(frequency_stats) > 0:
            print("✓ 频率分析功能正常")
        else:
            print("✗ 频率分析功能失败")
            return False
        
        # 测试热门冷门识别
        hot_numbers, cold_numbers = analyzer.identify_hot_cold_numbers(frequency_stats)
        if hot_numbers is not None and cold_numbers is not None:
            print("✓ 热门冷门识别功能正常")
        else:
            print("✗ 热门冷门识别功能失败")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 分析器测试失败: {e}")
        traceback.print_exc()
        return False

def test_visualizer():
    """测试可视化功能"""
    print("测试可视化功能...")
    try:
        visualizer = LotteryVisualizer()
        
        # 创建测试数据
        frequency_stats = {}
        for i in range(1, 50):
            frequency_stats[i] = {
                'frequency': i % 10,
                'frequency_rate': (i % 10) / 10,
                'last_appeared_period': i,
                'hot_score': (i % 10) / 10,
                'cold_score': 1 - (i % 10) / 10,
                'appearances': [i]
            }
        
        hot_numbers = [(i, i/10) for i in range(1, 11)]
        cold_numbers = [(i, i/10) for i in range(40, 50)]
        
        # 测试图表生成（不显示）
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        # 测试频率分析图
        try:
            visualizer.plot_frequency_analysis(frequency_stats, 'test_frequency.png')
            print("✓ 频率分析图生成正常")
        except Exception as e:
            print(f"✗ 频率分析图生成失败: {e}")
            return False
        
        # 测试热门冷门图
        try:
            visualizer.plot_hot_cold_numbers(hot_numbers, cold_numbers, 'test_hot_cold.png')
            print("✓ 热门冷门图生成正常")
        except Exception as e:
            print(f"✗ 热门冷门图生成失败: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 可视化测试失败: {e}")
        traceback.print_exc()
        return False

def test_complete_system():
    """测试完整系统"""
    print("测试完整系统...")
    try:
        system = MacauLotterySystem()
        
        # 生成测试数据
        success_count = system.generate_test_data(1, 10)
        if success_count > 0:
            print(f"✓ 测试数据生成成功: {success_count}期")
        else:
            print("✗ 测试数据生成失败")
            return False
        
        # 运行分析
        analysis_result = system.analyze_data(1, 10)
        if analysis_result:
            print("✓ 系统分析功能正常")
            print(f"  推荐号码: {analysis_result['recommendations']['comprehensive']}")
            print(f"  置信度: {analysis_result['confidence_score']}")
        else:
            print("✗ 系统分析功能失败")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 完整系统测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("澳门六合彩分析系统 - 功能测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("数据库功能", test_database),
        ("爬虫功能", test_crawler),
        ("分析器功能", test_analyzer),
        ("可视化功能", test_visualizer),
        ("完整系统", test_complete_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查系统配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)