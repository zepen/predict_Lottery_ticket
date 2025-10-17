#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从网站HTML中提取完整的生肖号码映射
"""

def extract_website_zodiac_mapping():
    """提取网站上的完整生肖号码映射"""
    
    # 从网站HTML中提取的完整生肖号码对应关系
    website_zodiac_numbers = {
        '鼠': [6, 18, 30, 42],      # 绿、红、红、蓝
        '牛': [5, 17, 29, 41],      # 绿、绿、红、蓝  
        '虎': [4, 16, 28, 40],      # 蓝、绿、绿、红
        '兔': [3, 15, 27, 39],      # 蓝、蓝、绿、绿
        '龙': [2, 14, 26, 38],      # 红、蓝、蓝、绿
        '蛇': [1, 13, 25, 37, 49],  # 红、红、蓝、蓝、绿
        '马': [12, 24, 36, 48],     # 红、红、蓝、蓝
        '羊': [11, 23, 35, 47],     # 绿、红、红、蓝
        '猴': [10, 22, 34, 46],     # 蓝、绿、红、红
        '鸡': [9, 21, 33, 45],      # 蓝、绿、绿、红
        '狗': [8, 20, 32, 44],      # 红、蓝、绿、绿
        '猪': [7, 19, 31, 43]       # 红、红、蓝、绿
    }
    
    # 网站上的波色号码
    website_colors = {
        '红波': [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46],
        '蓝波': [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48],
        '绿波': [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    }
    
    # 网站上的生肖波色分类
    website_zodiac_colors = {
        '红肖': ['马', '兔', '鼠', '鸡'],
        '蓝肖': ['蛇', '虎', '猪', '猴'],
        '绿肖': ['羊', '龙', '牛', '狗']
    }
    
    print("🌐 网站完整生肖号码对应关系")
    print("=" * 60)
    
    print("\n📊 网站生肖对应号码:")
    for zodiac, numbers in website_zodiac_numbers.items():
        print(f"{zodiac}: {numbers}")
    
    print("\n🔍 分析每个生肖的号码和波色:")
    for zodiac, numbers in website_zodiac_numbers.items():
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
    print("网站上的生肖号码对应关系确实与传统的不同！")
    print("需要根据网站的实际数据来修正生肖号码映射。")
    
    return website_zodiac_numbers

if __name__ == "__main__":
    extract_website_zodiac_mapping()