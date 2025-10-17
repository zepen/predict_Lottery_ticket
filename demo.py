#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
澳门六合彩分析系统演示脚本
展示系统的完整功能和分析结果
"""

import sys
import os
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MacauLotterySystem

def demo_complete_system():
    """演示完整系统功能"""
    print("🎰 澳门六合彩分析系统演示")
    print("=" * 60)
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 创建系统实例
    system = MacauLotterySystem()
    
    print("📊 步骤1: 生成测试数据 (模拟001-100期)")
    print("-" * 40)
    success_count = system.generate_test_data(1, 100)
    print(f"✅ 成功生成 {success_count} 期测试数据")
    print()
    
    print("🔍 步骤2: 运行深度分析")
    print("-" * 40)
    analysis_result = system.analyze_data(1, 100)
    
    if analysis_result:
        print("✅ 分析完成！")
        print()
        
        print("📈 分析结果摘要:")
        print(f"   分析期数: {analysis_result['period_range']}")
        print(f"   总期数: {analysis_result['total_periods']}")
        print(f"   置信度: {analysis_result['confidence_score']:.3f}")
        print()
        
        print("🔥 热门号码 Top 10:")
        for i, (num, score) in enumerate(analysis_result['hot_numbers'][:10], 1):
            print(f"   {i:2d}. 号码 {num:2d}: {score:.3f}")
        print()
        
        print("❄️  冷门号码 Top 10:")
        for i, (num, score) in enumerate(analysis_result['cold_numbers'][:10], 1):
            print(f"   {i:2d}. 号码 {num:2d}: {score:.3f}")
        print()
        
        print("🎯 推荐号码:")
        recommendations = analysis_result['recommendations']
        print(f"   热度推荐: {', '.join(map(str, recommendations['hot_based'][:6]))}")
        print(f"   模式推荐: {', '.join(map(str, recommendations['pattern_based'][:6]))}")
        print(f"   矩阵推荐: {', '.join(map(str, recommendations['matrix_based'][:6]))}")
        print(f"   综合推荐: {', '.join(map(str, recommendations['comprehensive']))}")
        print()
        
        print("📊 步骤3: 生成可视化图表")
        print("-" * 40)
        system.visualize_results(analysis_result)
        print("✅ 所有图表和报告已生成")
        print()
        
        print("📁 生成的文件:")
        print("   📈 频率分析图: reports/frequency_analysis_*.png")
        print("   🔥 热门冷门图: reports/hot_cold_numbers_*.png")
        print("   📊 模式分析图: reports/pattern_analysis_*.png")
        print("   🎛️  交互式仪表板: reports/interactive_dashboard_*.html")
        print("   📋 综合分析报告: reports/comprehensive_report_*.html")
        print()
        
        print("🎉 演示完成！系统运行正常。")
        
        return True
    else:
        print("❌ 分析失败")
        return False

def demo_advanced_features():
    """演示高级功能"""
    print("\n🚀 高级功能演示")
    print("=" * 60)
    
    system = MacauLotterySystem()
    
    print("📊 1. 数据库查询演示")
    print("-" * 30)
    
    # 查询数据库中的记录
    df = system.db.get_lottery_records(1, 10)
    if not df.empty:
        print(f"   数据库中共有 {len(df)} 期记录")
        print("   最近5期开奖记录:")
        for _, row in df.tail(5).iterrows():
            print(f"   第{row['period']:3d}期: {row['numbers']} (日期: {row['draw_date']})")
    print()
    
    print("🔍 2. 频率统计演示")
    print("-" * 30)
    
    # 获取频率统计
    freq_df = system.db.get_number_frequency()
    if not freq_df.empty:
        print("   号码频率统计 (前10个):")
        for _, row in freq_df.head(10).iterrows():
            print(f"   号码 {int(row['number']):2d}: 出现{int(row['frequency']):2d}次, 热度{row['hot_score']:.3f}")
    print()
    
    print("📈 3. 分析历史演示")
    print("-" * 30)
    
    # 获取分析结果
    analysis_df = system.db.get_analysis_results()
    if not analysis_df.empty:
        print(f"   共有 {len(analysis_df)} 条分析记录")
        print("   最新分析结果:")
        latest = analysis_df.iloc[0]
        print(f"   期数: {latest['period']}")
        print(f"   分析类型: {latest['analysis_type']}")
        print(f"   推荐号码: {latest['recommended_numbers']}")
        print(f"   置信度: {latest['confidence_score']:.3f}")
    print()

def main():
    """主演示函数"""
    try:
        # 基础功能演示
        success = demo_complete_system()
        
        if success:
            # 高级功能演示
            demo_advanced_features()
            
            print("\n" + "=" * 60)
            print("🎊 澳门六合彩分析系统演示完成！")
            print("=" * 60)
            print()
            print("💡 使用提示:")
            print("   1. 运行 'python3 main.py --test' 使用测试数据")
            print("   2. 运行 'python3 main.py --start 1 --end 289' 分析真实数据")
            print("   3. 运行 'python3 main.py --auto-update' 启动自动更新")
            print("   4. 查看 reports/ 目录获取详细分析报告")
            print()
            print("⚠️  免责声明: 本系统仅供学习和研究使用，预测结果不保证准确性")
        else:
            print("❌ 演示失败，请检查系统配置")
            return 1
            
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())