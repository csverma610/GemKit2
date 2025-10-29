"""
This module defines the Task dataclass for managing tasks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Enum for task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Enum for task priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Task:
    """Data class to represent a task"""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def mark_in_progress(self):
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = datetime.now()

    def mark_completed(self):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_failed(self, error: str):
        """Mark task as failed with error message"""
        self.status = TaskStatus.FAILED
        self.error = error
        self.updated_at = datetime.now()

    def mark_cancelled(self):
        """Mark task as cancelled"""
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'tags': self.tags,
            'assigned_to': self.assigned_to,
            'metadata': self.metadata,
            'error': self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            status=TaskStatus(data.get('status', 'pending')),
            priority=TaskPriority(data.get('priority', 'medium')),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now(),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            tags=data.get('tags', []),
            assigned_to=data.get('assigned_to'),
            metadata=data.get('metadata', {}),
            error=data.get('error')
        )

    def __repr__(self) -> str:
        """String representation of task"""
        return (f"Task(id='{self.id}', title='{self.title}', "
                f"status={self.status.value}, priority={self.priority.value})")
