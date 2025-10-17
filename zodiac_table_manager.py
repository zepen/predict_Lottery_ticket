#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六合彩生肖属性表管理器
保存和管理六合彩的生肖属性表内容
"""

import pandas as pd
import json
import sqlite3
from datetime import datetime
import requests
from bs4 import BeautifulSoup

class ZodiacTableManager:
    def __init__(self):
        self.zodiac_mapping = self.create_zodiac_mapping()
        self.db_path = "zodiac_table.db"
        self.init_database()
    
    def create_zodiac_mapping(self):
        """创建完整的生肖属性表"""
        # 六合彩生肖对应表（基于传统生肖轮换）
        zodiac_mapping = {}
        
        # 生肖顺序：鼠、牛、虎、兔、龙、蛇、马、羊、猴、鸡、狗、猪
        zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        # 每个生肖对应的号码范围（基于传统六合彩生肖表）
        zodiac_numbers = {
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
        
        # 创建号码到生肖的映射
        for zodiac, numbers in zodiac_numbers.items():
            for num in numbers:
                zodiac_mapping[num] = zodiac
        
        return zodiac_mapping
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建生肖属性表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zodiac_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number INTEGER UNIQUE NOT NULL,
                zodiac TEXT NOT NULL,
                zodiac_index INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建生肖统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zodiac_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zodiac TEXT UNIQUE NOT NULL,
                numbers TEXT NOT NULL,
                count INTEGER NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_zodiac_table(self):
        """保存生肖属性表到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清空现有数据
        cursor.execute('DELETE FROM zodiac_table')
        
        # 插入新的生肖属性表
        zodiac_order = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        for number, zodiac in self.zodiac_mapping.items():
            zodiac_index = zodiac_order.index(zodiac)
            cursor.execute('''
                INSERT INTO zodiac_table (number, zodiac, zodiac_index)
                VALUES (?, ?, ?)
            ''', (number, zodiac, zodiac_index))
        
        # 保存生肖统计信息
        cursor.execute('DELETE FROM zodiac_stats')
        
        zodiac_numbers = {}
        for number, zodiac in self.zodiac_mapping.items():
            if zodiac not in zodiac_numbers:
                zodiac_numbers[zodiac] = []
            zodiac_numbers[zodiac].append(number)
        
        for zodiac, numbers in zodiac_numbers.items():
            zodiac_index = zodiac_order.index(zodiac)
            description = f"生肖{zodiac}对应的号码范围"
            
            cursor.execute('''
                INSERT INTO zodiac_stats (zodiac, numbers, count, description)
                VALUES (?, ?, ?, ?)
            ''', (zodiac, ','.join(map(str, numbers)), len(numbers), description))
        
        conn.commit()
        conn.close()
        
        print("✅ 生肖属性表已保存到数据库")
    
    def export_zodiac_table_to_excel(self, filename="zodiac_table.xlsx"):
        """导出生肖属性表到Excel"""
        # 创建DataFrame
        data = []
        zodiac_order = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        for number in range(1, 50):
            zodiac = self.zodiac_mapping.get(number, '未知')
            zodiac_index = zodiac_order.index(zodiac) if zodiac in zodiac_order else -1
            
            data.append({
                '号码': number,
                '生肖': zodiac,
                '生肖序号': zodiac_index + 1,
                '生肖描述': f"第{zodiac_index + 1}个生肖" if zodiac_index >= 0 else "未知生肖"
            })
        
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"✅ 生肖属性表已导出到: {filename}")
        return filename
    
    def export_zodiac_table_to_json(self, filename="zodiac_table.json"):
        """导出生肖属性表到JSON"""
        zodiac_data = {
            'table_name': '六合彩生肖属性表',
            'description': '基于传统生肖轮换的六合彩号码与生肖对应关系',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'zodiac_mapping': self.zodiac_mapping,
            'zodiac_numbers': self.get_zodiac_numbers_mapping(),
            'zodiac_order': ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(zodiac_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 生肖属性表已导出到: {filename}")
        return filename
    
    def get_zodiac_numbers_mapping(self):
        """获取生肖到号码的映射"""
        zodiac_numbers = {}
        for number, zodiac in self.zodiac_mapping.items():
            if zodiac not in zodiac_numbers:
                zodiac_numbers[zodiac] = []
            zodiac_numbers[zodiac].append(number)
        
        # 对每个生肖的号码进行排序
        for zodiac in zodiac_numbers:
            zodiac_numbers[zodiac].sort()
        
        return zodiac_numbers
    
    def print_zodiac_table(self):
        """打印生肖属性表"""
        print("🐉 六合彩生肖属性表")
        print("=" * 60)
        
        zodiac_order = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        
        for i, zodiac in enumerate(zodiac_order, 1):
            numbers = [num for num, z in self.zodiac_mapping.items() if z == zodiac]
            numbers.sort()
            print(f"{i:2d}. {zodiac}: {', '.join(map(str, numbers))}")
        
        print("=" * 60)
        print(f"总计: {len(self.zodiac_mapping)} 个号码对应 12 个生肖")
    
    def get_zodiac_by_number(self, number):
        """根据号码获取生肖"""
        return self.zodiac_mapping.get(number, '未知')
    
    def get_numbers_by_zodiac(self, zodiac):
        """根据生肖获取号码"""
        return [num for num, z in self.zodiac_mapping.items() if z == zodiac]
    
    def analyze_zodiac_distribution(self):
        """分析生肖分布"""
        zodiac_counts = {}
        for zodiac in self.zodiac_mapping.values():
            zodiac_counts[zodiac] = zodiac_counts.get(zodiac, 0) + 1
        
        print("📊 生肖分布统计:")
        print("-" * 30)
        for zodiac, count in sorted(zodiac_counts.items()):
            print(f"{zodiac}: {count} 个号码")
        
        return zodiac_counts
    
    def create_zodiac_visualization(self):
        """创建生肖属性表可视化"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        zodiac_order = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        zodiac_counts = [len(self.get_numbers_by_zodiac(z)) for z in zodiac_order]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        bars = ax1.bar(zodiac_order, zodiac_counts, color='skyblue', alpha=0.7)
        ax1.set_title('六合彩生肖号码分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('生肖', fontsize=12)
        ax1.set_ylabel('号码数量', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, zodiac_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 饼图
        ax2.pie(zodiac_counts, labels=zodiac_order, autopct='%1.1f%%', startangle=90)
        ax2.set_title('生肖分布比例', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('zodiac_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ 生肖分布图已保存: zodiac_distribution.png")
        plt.show()
    
    def run_complete_zodiac_table_management(self):
        """运行完整的生肖属性表管理"""
        print("🐉 六合彩生肖属性表管理系统")
        print("=" * 60)
        print(f"管理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 打印生肖属性表
        self.print_zodiac_table()
        print()
        
        # 2. 分析生肖分布
        zodiac_counts = self.analyze_zodiac_distribution()
        print()
        
        # 3. 保存到数据库
        print("💾 保存生肖属性表到数据库...")
        self.save_zodiac_table()
        print()
        
        # 4. 导出到Excel
        print("📊 导出生肖属性表到Excel...")
        excel_file = self.export_zodiac_table_to_excel()
        print()
        
        # 5. 导出到JSON
        print("📄 导出生肖属性表到JSON...")
        json_file = self.export_zodiac_table_to_json()
        print()
        
        # 6. 创建可视化
        print("📈 创建生肖分布可视化...")
        self.create_zodiac_visualization()
        print()
        
        print("🎉 生肖属性表管理完成！")
        print(f"生成文件:")
        print(f"  - 数据库: {self.db_path}")
        print(f"  - Excel: {excel_file}")
        print(f"  - JSON: {json_file}")
        print(f"  - 图表: zodiac_distribution.png")

def main():
    """主函数"""
    manager = ZodiacTableManager()
    manager.run_complete_zodiac_table_management()

if __name__ == "__main__":
    main()