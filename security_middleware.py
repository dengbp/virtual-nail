#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import logging
import sys
from functools import wraps
from flask import request, abort, current_app

def sanitize_string(text):
    """
    清理字符串，移除或替换危险字符
    """
    if not isinstance(text, str):
        return str(text)
    
    # 移除或替换无法用UTF-8编码的字符
    try:
        # 尝试编码为UTF-8
        text.encode('utf-8')
    except UnicodeEncodeError:
        # 如果失败，使用replace模式
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    # 移除控制字符（除了常见的空白字符）
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # 移除可能的SQL注入字符
    dangerous_patterns = [
        r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
        r'(\b(or|and)\b\s+\d+\s*=\s*\d+)',
        r'(\b(exec|execute|script)\b)',
        r'(\b(javascript|vbscript|onload|onerror)\b)',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '[BLOCKED]', text, flags=re.IGNORECASE)
    
    return text

def is_malicious_request(request_data):
    """
    检测恶意请求
    """
    if not request_data:
        return False
    
    # 检查是否包含恶意字符
    malicious_chars = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0B', '\x0C', '\x0E', '\x0F', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1A',
        '\x1B', '\x1C', '\x1D', '\x1E', '\x1F', '\x7F'
    ]
    
    request_str = str(request_data)
    for char in malicious_chars:
        if char in request_str:
            return True
    
    # 检查是否包含明显的恶意模式
    malicious_patterns = [
        r'\x16\x03\x01',  # SSL/TLS握手
        r'\x00\x00\x00',  # 空字节序列
        r'\.\./',         # 路径遍历
        r'%00',           # URL编码的空字节
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, request_str, re.IGNORECASE):
            return True
    
    return False

def safe_log_request(logger, message, *args, **kwargs):
    """
    安全地记录请求日志
    """
    try:
        # 清理消息
        if args:
            cleaned_args = [sanitize_string(arg) for arg in args]
            message = message % tuple(cleaned_args)
        else:
            message = sanitize_string(message)
        
        logger.info(message, **kwargs)
    except Exception as e:
        # 如果记录失败，使用备用方案
        fallback_msg = f"[LOG ERROR] Failed to log original message: {str(e)}"
        try:
            logger.warning(fallback_msg)
        except:
            # 最后的备用方案 - 直接打印到控制台
            print(f"CRITICAL: {fallback_msg}")

def security_middleware(app):
    """
    Flask安全中间件
    """
    @app.before_request
    def before_request():
        # 检查请求数据
        request_data = {
            'url': request.url,
            'method': request.method,
            'headers': dict(request.headers),
            'args': dict(request.args),
            'form': dict(request.form),
            'data': request.get_data(as_text=True)
        }
        
        # 检测恶意请求
        if is_malicious_request(request_data):
            # 记录恶意请求（但不包含原始数据）
            safe_log_request(
                current_app.logger,
                "Malicious request detected from %s - %s %s",
                request.remote_addr,
                request.method,
                request.path
            )
            # 返回403禁止访问
            abort(403, description="Forbidden")
        
        # 清理请求数据用于日志记录
        cleaned_data = {}
        for key, value in request_data.items():
            if isinstance(value, dict):
                cleaned_data[key] = {k: sanitize_string(v) for k, v in value.items()}
            else:
                cleaned_data[key] = sanitize_string(value)
        
        # 安全地记录请求
        safe_log_request(
            current_app.logger,
            "Request: %s %s from %s",
            request.method,
            request.path,
            request.remote_addr
        )

def secure_logging_decorator(func):
    """
    装饰器：为函数添加安全的日志记录
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        try:
            # 记录函数调用
            safe_log_request(logger, f"Calling {func.__name__}")
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录成功
            safe_log_request(logger, f"Function {func.__name__} completed successfully")
            
            return result
            
        except Exception as e:
            # 记录错误
            safe_log_request(logger, f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

# 配置安全的日志处理器
def setup_secure_logging(app, log_file='secure_app.log'):
    """
    为Flask应用设置安全的日志配置
    """
    import os
    from logging.handlers import RotatingFileHandler
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置应用日志
    if not app.debug:
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到应用
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
    
    # 设置werkzeug日志级别
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    return app 