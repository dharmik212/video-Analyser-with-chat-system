"""
Conversation memory management
"""

from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger('video_chat.memory')


class ConversationMemory:
    """Manage conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        self.video_context = None
    
    def set_video_context(self, video_path: str, frames: List[str], initial_description: str = None):
        """Set the current video being discussed."""
        self.video_context = {
            "video_path": video_path,
            "frames": [str(f) for f in frames],
            "loaded_at": datetime.now().isoformat(),
            "initial_description": initial_description
        }
        logger.info(f"📹 Video context set: {Path(video_path).name}")
    
    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to history."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        }
        
        self.history.append(exchange)
        
        # Keep only last N exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"💬 Added exchange (total: {len(self.history)})")
    
    def get_context_summary(self) -> str:
        """Get formatted context for the model."""
        if not self.video_context:
            return "No video loaded."
        
        context = [f"Video: {Path(self.video_context['video_path']).name}"]
        
        if self.video_context.get('initial_description'):
            context.append(f"Initial analysis: {self.video_context['initial_description']}")
        
        if self.history:
            context.append("\nPrevious conversation:")
            for ex in self.history[-3:]:  # Last 3 exchanges
                context.append(f"Q: {ex['question']}")
                context.append(f"A: {ex['answer']}")
        
        return "\n".join(context)
    
    def save_conversation(self, output_path: str = None):
        """Save conversation to JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/conversation_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "video_context": self.video_context,
            "history": self.history,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Conversation saved: {output_path}")
        return output_path
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.video_context = None
        logger.info("🗑️  Conversation cleared")
