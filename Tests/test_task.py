import unittest
from datetime import datetime
from gemkit.task import Task, TaskStatus, TaskPriority

class TestTask(unittest.TestCase):
    """
    Unit tests for the Task class.
    """

    def test_task_creation(self):
        """
        Test that a Task object is created with the correct default values.
        """
        task = Task(id="1", title="Test Task", description="This is a test task.")
        self.assertEqual(task.id, "1")
        self.assertEqual(task.title, "Test Task")
        self.assertEqual(task.description, "This is a test task.")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.priority, TaskPriority.MEDIUM)

    def test_status_changes(self):
        """
        Test that the status of a task can be changed correctly.
        """
        task = Task(id="1", title="Test Task", description="This is a test task.")
        task.mark_in_progress()
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        task.mark_completed()
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.completed_at)
        task.mark_failed("Test error")
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.error, "Test error")
        task.mark_cancelled()
        self.assertEqual(task.status, TaskStatus.CANCELLED)

    def test_to_dict_and_from_dict(self):
        """
        Test that a Task object can be correctly converted to and from a dictionary.
        """
        task = Task(id="1", title="Test Task", description="This is a test task.")
        task_dict = task.to_dict()
        new_task = Task.from_dict(task_dict)
        self.assertEqual(task, new_task)

if __name__ == '__main__':
    unittest.main()
