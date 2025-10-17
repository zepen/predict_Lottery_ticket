#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析网站上的实际波色和生肖对应关系
"""

def analyze_website_mappings():
    """分析网站上的实际映射关系"""
    
    # 网站上的波色号码
    website_colors = {
        '红波': [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46],
        '蓝波': [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48],
        '绿波': [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    }
    
    # 网站上的生肖波色
    website_zodiac_colors = {
        '红肖': ['马', '兔', '鼠', '鸡'],
        '蓝肖': ['蛇', '虎', '猪', '猴'],
        '绿肖': ['羊', '龙', '牛', '狗']
    }
    
    # 传统生肖号码（每12个号码一组）
    traditional_zodiac_numbers = {
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
    
    print("🌐 网站实际映射关系分析")
    print("=" * 60)
    
    print("\n📊 网站波色号码分布:")
    for color, numbers in website_colors.items():
        print(f"{color}: {numbers} ({len(numbers)}个)")
    
    print("\n🐉 网站生肖波色分布:")
    for color, zodiacs in website_zodiac_colors.items():
        print(f"{color}: {zodiacs}")
    
    print("\n🔍 分析每个生肖的号码和波色:")
    for zodiac, numbers in traditional_zodiac_numbers.items():
        # 找出每个生肖号码对应的波色
        colors = []
        for num in numbers:
            for color, color_numbers in website_colors.items():
                if num in color_numbers:
                    colors.append(color)
                    break
        
        # 找出生肖的波色分类
        zodiac_color = None
        for color, zodiacs in website_zodiac_colors.items():
            if zodiac in zodiacs:
                zodiac_color = color
                break
        
        print(f"\n{zodiac}:")
        print(f"  号码: {numbers}")
        print(f"  号码波色: {colors}")
        print(f"  生肖波色: {zodiac_color}")
        
        # 检查一致性
        unique_colors = set(colors)
        if len(unique_colors) == 1 and zodiac_color:
            expected_color = zodiac_color.replace('肖', '波')
            actual_color = list(unique_colors)[0]
            if expected_color == actual_color:
                print(f"  ✅ 一致: {actual_color}")
            else:
                print(f"  ❌ 不一致: 期望{expected_color}, 实际{actual_color}")
        else:
            print(f"  ❌ 号码波色不一致: {unique_colors}")
    
    print("\n" + "=" * 60)
    print("💡 结论:")
    print("网站上的波色和生肖对应关系与传统的不同！")
    print("需要根据网站的实际数据来修正映射关系。")

if __name__ == "__main__":
    analyze_website_mappings()