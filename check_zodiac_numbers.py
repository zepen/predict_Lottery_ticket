#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查网站上的生肖对应号码
"""

def analyze_website_zodiac_numbers():
    """分析网站上的生肖号码对应关系"""
    
    # 从网站HTML中提取的生肖对应号码
    website_zodiac_numbers = {
        '鼠': 6,   # bg_green
        '牛': 5,   # bg_green  
        '虎': 4,   # bg_blue
        '兔': 3,   # bg_blue
        '龙': 2,   # bg_red
        '蛇': 1,   # bg_red
        '马': 12,  # bg_red
        '羊': 11,  # bg_green
        '猴': 10,  # bg_blue
        '鸡': 9,   # bg_blue
        '狗': 8,   # bg_red
        '猪': 7    # bg_red
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
    
    print("🌐 网站生肖号码对应关系分析")
    print("=" * 60)
    
    print("\n📊 网站生肖对应号码:")
    for zodiac, number in website_zodiac_numbers.items():
        print(f"{zodiac}: {number:2d}")
    
    print("\n🔍 分析每个生肖的号码和波色:")
    for zodiac, number in website_zodiac_numbers.items():
        # 找出生肖号码对应的波色
        number_color = None
        for color, numbers in website_colors.items():
            if number in numbers:
                number_color = color
                break
        
        # 找出生肖的波色分类
        zodiac_color = None
        for color, zodiacs in website_zodiac_colors.items():
            if zodiac in zodiacs:
                zodiac_color = color
                break
        
        print(f"\n{zodiac}:")
        print(f"  号码: {number}")
        print(f"  号码波色: {number_color}")
        print(f"  生肖波色: {zodiac_color}")
        
        # 检查一致性
        if number_color and zodiac_color:
            expected_color = zodiac_color.replace('肖', '波')
            if expected_color == number_color:
                print(f"  ✅ 一致: {number_color}")
            else:
                print(f"  ❌ 不一致: 期望{expected_color}, 实际{number_color}")
        else:
            print(f"  ❌ 无法确定波色")
    
    print("\n" + "=" * 60)
    print("💡 结论:")
    print("网站上的生肖号码对应关系与传统的12生肖循环不同！")
    print("需要根据网站的实际数据来修正生肖号码映射。")

if __name__ == "__main__":
    analyze_website_zodiac_numbers()