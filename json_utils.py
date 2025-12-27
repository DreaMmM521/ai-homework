# json_utils.py
import json
import numpy as np
import pandas as pd
from datetime import datetime

class UniversalEncoder(json.JSONEncoder):
    """通用的JSON编码器，处理各种类型"""
    
    def default(self, obj):
        # 处理numpy类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # 处理pandas类型
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        
        # 处理datetime类型
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        
        # 处理其他可序列化类型
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        
        # 最后尝试使用str
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)