# convert_log_to_csv.py

import json
import csv
import uuid

# نام فایل‌های ورودی و خروجی
JSON_LOG_FILE = r"C:\Users\m.yaghoubi\Desktop\rag_core_service\monitoring_log.jsonl"
CSV_LOG_FILE = "qa_log.csv"

# هدرهای مورد نظر برای فایل CSV
CSV_HEADERS = [
    "log_id",
    "timestamp",
    "session_id",
    "question",
    "answer",
    "sources",
    "latency_ms"
]

def format_sources(source_pages):
    """
    لیست شماره صفحات را به فرمت رشته‌ای مورد نظر تبدیل می‌کند.
    مثال: [7, 9, 15] -> '["Page 7", "Page 9", "Page 15"]'
    """
    if not source_pages:
        return "[]"
    
    # برای مطابقت با فرمت نمونه، می‌توانیم "Page" را اضافه کنیم یا فقط شماره‌ها را بگذاریم
    # در اینجا برای سادگی فقط شماره‌ها را قرار می‌دهیم
    formatted = [f'"{page}"' for page in source_pages]
    return f'[{", ".join(formatted)}]'


def convert_json_to_csv():
    """
    فایل monitoring_log.jsonl را می‌خواند و آن را به qa_log.csv تبدیل می‌کند.
    """
    print(f"Reading logs from '{JSON_LOG_FILE}'...")
    
    try:
        with open(JSON_LOG_FILE, 'r', encoding='utf-8') as json_file, \
             open(CSV_LOG_FILE, 'w', newline='', encoding='utf-8') as csv_file:

            writer = csv.writer(csv_file)
            
            # نوشتن هدر فایل CSV
            writer.writerow(CSV_HEADERS)

            # خواندن هر خط از فایل JSONL
            for line in json_file:
                try:
                    log_data = json.loads(line)

                    # استخراج و تبدیل داده‌ها به فرمت مورد نظر
                    log_id = str(uuid.uuid4()) # تولید یک ID جدید برای هر لاگ
                    timestamp = log_data.get("asctime", "")
                    session_id = log_data.get("session_id", "")
                    question = log_data.get("query", "")
                    
                    # از پاسخ خام LLM استفاده می‌کنیم که تمیزتر است
                    answer = log_data.get("llm_answer", "") 
                    
                    source_pages = log_data.get("source_pages", [])
                    sources_str = format_sources(source_pages)
                    
                    # تبدیل ثانیه به میلی‌ثانیه
                    latency_ms = log_data.get("response_time_seconds", 0) * 1000

                    # نوشتن سطر جدید در فایل CSV
                    writer.writerow([
                        log_id,
                        timestamp,
                        session_id,
                        question,
                        answer,
                        sources_str,
                        latency_ms
                    ])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Skipping line due to missing key {e}: {line.strip()}")

        print(f"Successfully converted logs to '{CSV_LOG_FILE}'")

    except FileNotFoundError:
        print(f"Error: Input file '{JSON_LOG_FILE}' not found.")


if __name__ == "__main__":
    convert_json_to_csv()