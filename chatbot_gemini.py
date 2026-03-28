#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatbot sử dụng Google Gemini để trả lời câu hỏi về AQI và sức khỏe
Tích hợp với Streamlit
"""

import google.generativeai as genai
import sqlite3
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatbotGemini:
    def __init__(self, api_key=None, db_path="aqi_analysis.db"):
        """
        Khởi tạo chatbot Gemini
        
        Args:
            api_key: API key của Google Gemini (hoặc load từ biến môi trường)
            db_path: Đường dẫn đến database
        """
        self.db_path = db_path
        
        # Lấy API key từ biến môi trường hoặc tham số
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key is None:
            raise ValueError("⚠️ Cần cung cấp GEMINI_API_KEY. Cài đặt: set GEMINI_API_KEY=your_api_key")
        
        # Cấu hình Gemini
        genai.configure(api_key=api_key)
        
        # Model configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Tạo model - Sử dụng model mới nhất có sẵn
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # Model mới nhất, nhanh và hiệu quả
            generation_config=generation_config
        )
        
        # Conversation history
        self.conversation_history = []
        
    def get_db_data_summary(self):
        """Lấy tổng quan dữ liệu từ database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            aqi_count = cursor.execute('SELECT COUNT(*) FROM aqi_data').fetchone()[0]
            health_count = cursor.execute('SELECT COUNT(*) FROM health_data').fetchone()[0]
            avg_aqi = cursor.execute('SELECT AVG(aqi) FROM aqi_data').fetchone()[0]
            recent_aqi = pd.read_sql_query("""
                SELECT year, AVG(aqi) as avg_aqi, MAX(aqi) as max_aqi, MIN(aqi) as min_aqi
                FROM aqi_data 
                WHERE year >= 2020
                GROUP BY year
                ORDER BY year DESC
            """, conn)
            avg_health = cursor.execute('SELECT AVG(health_index) FROM health_data').fetchone()[0]
            
            conn.close()
            
            summary = {
                'total_aqi_records': aqi_count,
                'total_health_records': health_count,
                'avg_aqi': round(avg_aqi, 2),
                'avg_health_index': round(avg_health, 2),
                'recent_years': recent_aqi.to_dict('records')
            }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def query_database(self, query_type, params=None):
        """
        Truy vấn database dựa trên loại câu hỏi
        
        Args:
            query_type: Loại truy vấn (trends, seasonal, health, pollution)
            params: Tham số truy vấn
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if query_type == "recent_trends":
                # Xu hướng gần đây
                df = pd.read_sql_query("""
                    SELECT year, AVG(aqi) as avg_aqi, AVG(health_index) as avg_health
                    FROM aqi_data a
                    JOIN health_data h ON a.date = h.date
                    WHERE a.year >= 2020
                    GROUP BY year
                    ORDER BY year
                """, conn)
                
            elif query_type == "seasonal_analysis":
                # Phân tích theo mùa
                df = pd.read_sql_query("""
                    SELECT a.season,
                           AVG(a.aqi) as avg_aqi,
                           AVG(h.respiratory_cases) as avg_respiratory,
                           AVG(h.cardiovascular_cases) as avg_cardio,
                           AVG(h.health_index) as avg_health
                    FROM aqi_data a
                    JOIN health_data h ON a.date = h.date
                    GROUP BY a.season
                    ORDER BY AVG(a.aqi) DESC
                """, conn)
                
            elif query_type == "pollution_levels":
                # Mức độ ô nhiễm
                df = pd.read_sql_query("""
                    SELECT a.aqi_category,
                           COUNT(*) as days,
                           AVG(a.aqi) as avg_aqi,
                           AVG(h.total_cases) as avg_cases
                    FROM aqi_data a
                    JOIN health_data h ON a.date = h.date
                    GROUP BY a.aqi_category
                    ORDER BY AVG(a.aqi)
                """, conn)
                
            elif query_type == "health_impact":
                # Tác động sức khỏe
                df = pd.read_sql_query("""
                    SELECT 
                        CASE 
                            WHEN a.aqi <= 50 THEN 'Good'
                            WHEN a.aqi <= 100 THEN 'Moderate'
                            WHEN a.aqi <= 150 THEN 'Unhealthy for Sensitive'
                            WHEN a.aqi <= 200 THEN 'Unhealthy'
                            ELSE 'Very Unhealthy'
                        END as pollution_level,
                        AVG(h.respiratory_cases) as respiratory,
                        AVG(h.cardiovascular_cases) as cardiovascular,
                        AVG(h.total_cases) as total,
                        AVG(h.health_index) as health_index,
                        COUNT(*) as days
                    FROM aqi_data a
                    JOIN health_data h ON a.date = h.date
                    GROUP BY pollution_level
                    ORDER BY AVG(a.aqi)
                """, conn)
            
            else:
                return None
            
            conn.close()
            return df.to_dict('records')
            
        except Exception as e:
            return None
    
    def generate_response(self, user_question, use_data=True):
        """
        Tạo phản hồi từ chatbot
        
        Args:
            user_question: Câu hỏi của người dùng
            use_data: Có sử dụng dữ liệu database không
        """
        try:
            # Lấy dữ liệu nếu cần
            context_data = ""
            if use_data:
                summary = self.get_db_data_summary()
                context_data = f"""
BẠN CÓ DỮ LIỆU VỀ AQI VÀ SỨC KHỎE:
- Tổng số bản ghi AQI: {summary.get('total_aqi_records', 0):,}
- Tổng số bản ghi sức khỏe: {summary.get('total_health_records', 0):,}
- AQI trung bình: {summary.get('avg_aqi', 0):.1f}
- Chỉ số sức khỏe trung bình: {summary.get('avg_health_index', 0):.1f}
- Các năm gần đây (2020-2025):
{summary.get('recent_years', [])}

HOẶC BẠN CÓ THỂ TRUY VẤN DATABASE:
- Xu hướng AQI theo năm
- Phân tích theo mùa
- Mức độ ô nhiễm
- Tác động sức khỏe
                """
            
            # Prompt cho Gemini
            prompt = f"""
BẠN LÀ MỘT CHUYÊN GIA VỀ CHẤT LƯỢNG KHÔNG KHÍ (AQI) VÀ SỨC KHỎE CỘNG ĐỒNG TẠI VIỆT NAM.

{context_data}

HÃY TRẢ LỜI CÂU HỎI SAU ĐÂY BẰNG TIẾNG VIỆT, RÕ RÀNG, CHÍNH XÁC VÀ DỄ HIỂU:

CÂU HỎI: {user_question}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời bằng tiếng Việt
- Nếu được, sử dụng dữ liệu thực tế
- Đưa ra khuyến nghị cụ thể
- Giải thích dễ hiểu
- Nếu không có đủ dữ liệu, hãy giải thích lý do

HÃY BẮT ĐẦU TRẢ LỜI:
"""
            
            # Gọi API Gemini
            response = self.model.generate_content(prompt)
            
            # Lưu lịch sử
            self.conversation_history.append({
                'user': user_question,
                'assistant': response.text,
                'timestamp': datetime.now()
            })
            
            return response.text
            
        except Exception as e:
            return f"❌ Lỗi: {e}"
    
    def get_suggested_questions(self):
        """Lấy danh sách câu hỏi gợi ý"""
        return [
            "AQI là gì và ý nghĩa của nó?",
            "Chất lượng không khí hiện tại như thế nào?",
            "Ô nhiễm không khí ảnh hưởng đến sức khỏe như thế nào?",
            "Mùa nào có ô nhiễm cao nhất?",
            "PM2.5 là gì và tại sao nó nguy hiểm?",
            "Làm thế nào để bảo vệ bản thân khỏi ô nhiễm không khí?",
            "Xu hướng AQI trong những năm gần đây?",
            "Có mối liên hệ giữa AQI và bệnh hô hấp không?",
            "Ai là đối tượng dễ bị ảnh hưởng bởi ô nhiễm không khí?",
            "Khi nào nên hạn chế hoạt động ngoài trời?"
        ]
    
    def clear_history(self):
        """Xóa lịch sử hội thoại"""
        self.conversation_history = []


# Hàm helper để sử dụng trong Streamlit
def create_gemini_chatbot(api_key=None):
    """Tạo chatbot Gemini để sử dụng trong Streamlit"""
    return ChatbotGemini(api_key=api_key)


if __name__ == "__main__":
    # Test chatbot
    print("🤖 TESTING GEMINI CHATBOT")
    print("=" * 60)
    
    try:
        chatbot = ChatbotGemini()
        
        # Test câu hỏi
        test_questions = [
            "AQI là gì?",
            "Chất lượng không khí có cải thiện qua các năm không?",
            "Mùa nào ô nhiễm nhất?"
        ]
        
        for question in test_questions:
            print(f"\n👤 Câu hỏi: {question}")
            print(f"🤖 Trả lời: {chatbot.generate_response(question)}")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")

