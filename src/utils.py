"""
Utility functions for video chat system
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import colorlog

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # ═══════════════════════════════════════════════════════
    # FIX: Explicitly use UTF-8 encoding (Windows fix)
    # ═══════════════════════════════════════════════════════
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup colored logging."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger = colorlog.getLogger('video_chat')
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger


def format_conversation(history: list) -> str:
    """Format conversation history for display."""
    if not history:
        return "No conversation history yet."
    
    formatted = []
    for i, exchange in enumerate(history, 1):
        formatted.append(f"\n{'='*60}")
        formatted.append(f"[Exchange {i}]")
        formatted.append(f"👤 You: {exchange['question']}")
        formatted.append(f"🤖 Assistant: {exchange['answer']}")
    formatted.append(f"\n{'='*60}\n")
    return '\n'.join(formatted)
