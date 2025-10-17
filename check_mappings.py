#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查六合彩生肖、波色、号码对应关系是否正确
"""

def check_traditional_mappings():
    """检查传统六合彩映射关系"""
    
    # 传统六合彩生肖对应表
    zodiac_mapping = {
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
    
    # 传统六合彩波色对应表
    color_mapping = {
        '绿': [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46],
        '红': [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48],
        '蓝': [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    }
    
    print("🐉 六合彩生肖、波色、号码对应关系检查")
    print("=" * 60)
    
    # 检查每个生肖的号码和波色
    for zodiac, numbers in zodiac_mapping.items():
        print(f"\n{zodiac}:")
        print(f"  号码: {numbers}")
        
        # 检查每个号码的波色
        colors = []
        for num in numbers:
            for color, color_numbers in color_mapping.items():
                if num in color_numbers:
                    colors.append(color)
                    break
        
        print(f"  波色: {colors}")
        
        # 检查是否有重复或遗漏
        if len(set(colors)) == 1:
            print(f"  ✅ 波色一致: {colors[0]}")
        else:
            print(f"  ❌ 波色不一致: {colors}")
    
    print("\n" + "=" * 60)
    print("📊 波色统计:")
    
    # 统计每个波色的生肖分布
    for color, color_numbers in color_mapping.items():
        print(f"\n{color}波 ({len(color_numbers)}个号码):")
        zodiacs_in_color = []
        for zodiac, numbers in zodiac_mapping.items():
            common_numbers = set(numbers) & set(color_numbers)
            if common_numbers:
                zodiacs_in_color.append(f"{zodiac}({len(common_numbers)}个)")
        print(f"  包含生肖: {', '.join(zodiacs_in_color)}")
    
    print("\n" + "=" * 60)
    print("🔍 详细号码检查:")
    
    # 检查1-49号码的完整覆盖
    all_numbers = set()
    for numbers in zodiac_mapping.values():
        all_numbers.update(numbers)
    
    missing_numbers = set(range(1, 50)) - all_numbers
    if missing_numbers:
        print(f"❌ 遗漏号码: {sorted(missing_numbers)}")
    else:
        print("✅ 号码覆盖完整 (1-49)")
    
    # 检查重复号码
    all_numbers_list = []
    for numbers in zodiac_mapping.values():
        all_numbers_list.extend(numbers)
    
    duplicates = []
    for num in set(all_numbers_list):
        if all_numbers_list.count(num) > 1:
            duplicates.append(num)
    
    if duplicates:
        print(f"❌ 重复号码: {duplicates}")
    else:
        print("✅ 无重复号码")
    
    return zodiac_mapping, color_mapping

def check_alternative_mappings():
    """检查其他可能的映射方式"""
    print("\n" + "=" * 60)
    print("🔄 其他映射方式检查:")
    
    # 方式1：按生肖顺序每12个号码一组
    print("\n方式1 - 按生肖顺序每12个号码一组:")
    zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
    for i, zodiac in enumerate(zodiacs):
        numbers = []
        for j in range(5):  # 每个生肖5个号码
            num = i + 1 + j * 12
            if num <= 49:
                numbers.append(num)
        print(f"  {zodiac}: {numbers}")
    
    # 方式2：按波色分组
    print("\n方式2 - 按波色分组:")
    green = [1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46]
    red = [3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48]
    blue = [5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49]
    
    print(f"  绿波: {green} ({len(green)}个)")
    print(f"  红波: {red} ({len(red)}个)")
    print(f"  蓝波: {blue} ({len(blue)}个)")

if __name__ == "__main__":
    zodiac_mapping, color_mapping = check_traditional_mappings()
    check_alternative_mappings()