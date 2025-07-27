# app/monitoring.py

import logging
from pythonjsonlogger import jsonlogger

def setup_monitoring_logger():

    logger = logging.getLogger("RAG_Monitoring")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    log_handler = logging.FileHandler("monitoring_log.jsonl", mode='a')
    
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    return logger

monitoring_logger = setup_monitoring_logger()