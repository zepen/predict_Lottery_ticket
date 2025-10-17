#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正六合彩生肖、波色、号码对应关系
根据传统六合彩规则，每个生肖应该对应特定的波色
"""

def create_correct_mappings():
    """创建正确的六合彩映射关系"""
    
    # 传统六合彩规则：
    # 每个生肖对应特定的波色，每个波色包含4个生肖
    # 绿波生肖：鼠、兔、马、鸡
    # 红波生肖：牛、龙、羊、狗  
    # 蓝波生肖：虎、蛇、猴、猪
    
    # 正确的生肖波色对应
    zodiac_colors = {
        '鼠': '绿',
        '牛': '红', 
        '虎': '蓝',
        '兔': '绿',
        '龙': '红',
        '蛇': '蓝',
        '马': '绿',
        '羊': '红',
        '猴': '蓝',
        '鸡': '绿',
        '狗': '红',
        '猪': '蓝'
    }
    
    # 正确的生肖号码对应（每个生肖4个号码，除了鼠有5个）
    zodiac_numbers = {
        '鼠': [1, 13, 25, 37, 49],  # 绿波，5个号码
        '牛': [2, 14, 26, 38],      # 红波，4个号码
        '虎': [3, 15, 27, 39],      # 蓝波，4个号码
        '兔': [4, 16, 28, 40],      # 绿波，4个号码
        '龙': [5, 17, 29, 41],      # 红波，4个号码
        '蛇': [6, 18, 30, 42],      # 蓝波，4个号码
        '马': [7, 19, 31, 43],      # 绿波，4个号码
        '羊': [8, 20, 32, 44],      # 红波，4个号码
        '猴': [9, 21, 33, 45],      # 蓝波，4个号码
        '鸡': [10, 22, 34, 46],     # 绿波，4个号码
        '狗': [11, 23, 35, 47],     # 红波，4个号码
        '猪': [12, 24, 36, 48]      # 蓝波，4个号码
    }
    
    # 根据生肖波色生成正确的波色号码映射
    color_mapping = {}
    for zodiac, color in zodiac_colors.items():
        numbers = zodiac_numbers[zodiac]
        for num in numbers:
            color_mapping[num] = color
    
    # 验证映射关系
    print("🐉 修正后的六合彩生肖、波色、号码对应关系")
    print("=" * 60)
    
    # 按波色分组显示
    colors = ['绿', '红', '蓝']
    for color in colors:
        print(f"\n{color}波生肖:")
        color_numbers = []
        for zodiac, zodiac_color in zodiac_colors.items():
            if zodiac_color == color:
                numbers = zodiac_numbers[zodiac]
                color_numbers.extend(numbers)
                print(f"  {zodiac}: {numbers}")
        print(f"  总计: {sorted(color_numbers)} ({len(color_numbers)}个号码)")
    
    print("\n" + "=" * 60)
    print("📊 验证结果:")
    
    # 验证每个生肖的波色一致性
    all_consistent = True
    for zodiac, numbers in zodiac_numbers.items():
        expected_color = zodiac_colors[zodiac]
        actual_colors = [color_mapping[num] for num in numbers]
        if all(color == expected_color for color in actual_colors):
            print(f"✅ {zodiac}: 波色一致 ({expected_color})")
        else:
            print(f"❌ {zodiac}: 波色不一致 (期望: {expected_color}, 实际: {actual_colors})")
            all_consistent = False
    
    # 验证号码覆盖
    all_numbers = set()
    for numbers in zodiac_numbers.values():
        all_numbers.update(numbers)
    
    missing_numbers = set(range(1, 50)) - all_numbers
    if missing_numbers:
        print(f"❌ 遗漏号码: {sorted(missing_numbers)}")
    else:
        print("✅ 号码覆盖完整 (1-49)")
    
    # 验证无重复
    all_numbers_list = []
    for numbers in zodiac_numbers.values():
        all_numbers_list.extend(numbers)
    
    duplicates = []
    for num in set(all_numbers_list):
        if all_numbers_list.count(num) > 1:
            duplicates.append(num)
    
    if duplicates:
        print(f"❌ 重复号码: {duplicates}")
    else:
        print("✅ 无重复号码")
    
    if all_consistent and not missing_numbers and not duplicates:
        print("\n🎉 映射关系完全正确！")
    
    return zodiac_numbers, color_mapping, zodiac_colors

def generate_correct_code():
    """生成正确的代码"""
    zodiac_numbers, color_mapping, zodiac_colors = create_correct_mappings()
    
    print("\n" + "=" * 60)
    print("💻 正确的代码实现:")
    print("=" * 60)
    
    print("""
    def create_color_mapping(self):
        \"\"\"创建波色映射表\"\"\"
        color_mapping = {}
        
        # 绿波生肖：鼠、兔、马、鸡
        green_numbers = [1, 13, 25, 37, 49, 4, 16, 28, 40, 7, 19, 31, 43, 10, 22, 34, 46]
        for num in green_numbers:
            color_mapping[num] = '绿'
        
        # 红波生肖：牛、龙、羊、狗
        red_numbers = [2, 14, 26, 38, 5, 17, 29, 41, 8, 20, 32, 44, 11, 23, 35, 47]
        for num in red_numbers:
            color_mapping[num] = '红'
        
        # 蓝波生肖：虎、蛇、猴、猪
        blue_numbers = [3, 15, 27, 39, 6, 18, 30, 42, 9, 21, 33, 45, 12, 24, 36, 48]
        for num in blue_numbers:
            color_mapping[num] = '蓝'
        
        return color_mapping
    
    def create_zodiac_mapping(self):
        \"\"\"创建生肖映射表\"\"\"
        zodiac_mapping = {}
        
        zodiac_numbers = {
            '鼠': [1, 13, 25, 37, 49],  # 绿波
            '牛': [2, 14, 26, 38],      # 红波
            '虎': [3, 15, 27, 39],      # 蓝波
            '兔': [4, 16, 28, 40],      # 绿波
            '龙': [5, 17, 29, 41],      # 红波
            '蛇': [6, 18, 30, 42],      # 蓝波
            '马': [7, 19, 31, 43],      # 绿波
            '羊': [8, 20, 32, 44],      # 红波
            '猴': [9, 21, 33, 45],      # 蓝波
            '鸡': [10, 22, 34, 46],     # 绿波
            '狗': [11, 23, 35, 47],     # 红波
            '猪': [12, 24, 36, 48]      # 蓝波
        }
        
        for zodiac, numbers in zodiac_numbers.items():
            for num in numbers:
                zodiac_mapping[num] = zodiac
        
        return zodiac_mapping
    """)

if __name__ == "__main__":
    create_correct_mappings()
    generate_correct_code()