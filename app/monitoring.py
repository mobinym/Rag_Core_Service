# app/monitoring.py

import logging
from pythonjsonlogger import jsonlogger

def setup_monitoring_logger():
    """
    یک لاگر مجزا برای ثبت اطلاعات مانیتورینگ در یک فایل JSON راه‌اندازی می‌کند.
    """
    # ایجاد یک لاگر با نام مشخص
    logger = logging.getLogger("RAG_Monitoring")
    logger.setLevel(logging.INFO)
    
    # جلوگیری از ارسال لاگ‌ها به لاگر ریشه (جلوگیری از نمایش در کنسول)
    logger.propagate = False
    
    # ایجاد یک handler برای نوشتن لاگ‌ها در فایل
    log_handler = logging.FileHandler("monitoring_log.jsonl", mode='a')
    
    # تعریف فرمت JSON
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    return logger

# ایجاد یک نمونه از لاگر برای استفاده در کل برنامه
monitoring_logger = setup_monitoring_logger()