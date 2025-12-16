from .task import Task

class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def get_or_create(self, task_id, classes, **kwargs):
        if task_id not in self.tasks:
            self.tasks[task_id] = Task(task_id, classes, **kwargs)
        return self.tasks[task_id]
