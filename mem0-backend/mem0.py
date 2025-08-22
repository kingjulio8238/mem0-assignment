#!/usr/bin/env python3
"""
Mem0 - A simple memory storage and retrieval system for CLI usage
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

class Memory:
    def __init__(self, storage_file: str = "memories.json"):
        """Initialize the memory system with a JSON storage file"""
        self.storage_file = storage_file
        self.memories = self._load_memories()
    
    def _load_memories(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load memories from storage file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load {self.storage_file}, starting with empty memory")
                return {}
        return {}
    
    def _save_memories(self):
        """Save memories to storage file"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save to {self.storage_file}: {e}")
    
    def add(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Add a memory for a user"""
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        memory_entry = {
            "id": memory_id,
            "memory": message,
            "timestamp": timestamp,
            "event": "ADD"
        }
        
        self.memories[user_id].append(memory_entry)
        self._save_memories()
        
        return {
            "results": [memory_entry]
        }
    
    def search(self, query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
        """Search for memories matching the query"""
        if user_id not in self.memories:
            return {"results": []}
        
        query_lower = query.lower()
        matching_memories = []
        
        for memory in self.memories[user_id]:
            memory_text = memory.get("memory", "").lower()
            
            # Simple keyword matching
            if query_lower in memory_text:
                matching_memories.append(memory)
        
        # Sort by timestamp (newest first)
        matching_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit > 0:
            matching_memories = matching_memories[:limit]
        
        return {"results": matching_memories}
    
    def get_all(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Get all memories for a user"""
        return self.memories.get(user_id, [])
    
    def delete(self, memory_id: str, user_id: str = "default") -> bool:
        """Delete a specific memory"""
        if user_id not in self.memories:
            return False
        
        for i, memory in enumerate(self.memories[user_id]):
            if memory.get("id") == memory_id:
                del self.memories[user_id][i]
                self._save_memories()
                return True
        
        return False
    
    def clear_user(self, user_id: str = "default") -> bool:
        """Clear all memories for a user"""
        if user_id in self.memories:
            del self.memories[user_id]
            self._save_memories()
            return True
        return False
    
    def get_user_count(self, user_id: str = "default") -> int:
        """Get the number of memories for a user"""
        return len(self.memories.get(user_id, []))
    
    def list_users(self) -> List[str]:
        """List all users with memories"""
        return list(self.memories.keys())
