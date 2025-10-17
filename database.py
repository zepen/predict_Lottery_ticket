import sqlite3
import pandas as pd
from datetime import datetime
from config import DATABASE_PATH

class LotteryDatabase:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建开奖记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lottery_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period INTEGER UNIQUE NOT NULL,
                draw_date TEXT NOT NULL,
                numbers TEXT NOT NULL,
                special_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建号码频率统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS number_frequency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number INTEGER NOT NULL,
                frequency INTEGER DEFAULT 0,
                last_appeared_period INTEGER,
                hot_score REAL DEFAULT 0.0,
                cold_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                hot_numbers TEXT,
                cold_numbers TEXT,
                recommended_numbers TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_lottery_record(self, period, draw_date, numbers, special_number=None):
        """保存开奖记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO lottery_records 
                (period, draw_date, numbers, special_number)
                VALUES (?, ?, ?, ?)
            ''', (period, draw_date, numbers, special_number))
            conn.commit()
            return True
        except Exception as e:
            print(f"保存开奖记录失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_lottery_records(self, start_period=1, end_period=289):
        """获取开奖记录"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT period, draw_date, numbers, special_number
            FROM lottery_records
            WHERE period BETWEEN ? AND ?
            ORDER BY period
        '''
        df = pd.read_sql_query(query, conn, params=(start_period, end_period))
        conn.close()
        return df
    
    def update_number_frequency(self, number, frequency, last_period, hot_score, cold_score):
        """更新号码频率统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO number_frequency
            (number, frequency, last_appeared_period, hot_score, cold_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (number, frequency, last_period, hot_score, cold_score))
        
        conn.commit()
        conn.close()
    
    def get_number_frequency(self):
        """获取号码频率统计"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT number, frequency, last_appeared_period, hot_score, cold_score
            FROM number_frequency
            ORDER BY number
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def save_analysis_result(self, period, analysis_type, hot_numbers, cold_numbers, 
                           recommended_numbers, confidence_score):
        """保存分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results
            (period, analysis_type, hot_numbers, cold_numbers, recommended_numbers, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (period, analysis_type, hot_numbers, cold_numbers, recommended_numbers, confidence_score))
        
        conn.commit()
        conn.close()
    
    def get_analysis_results(self, period=None):
        """获取分析结果"""
        conn = sqlite3.connect(self.db_path)
        
        if period:
            query = '''
                SELECT * FROM analysis_results
                WHERE period = ?
                ORDER BY created_at DESC
            '''
            df = pd.read_sql_query(query, conn, params=(period,))
        else:
            query = '''
                SELECT * FROM analysis_results
                ORDER BY period DESC, created_at DESC
            '''
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df