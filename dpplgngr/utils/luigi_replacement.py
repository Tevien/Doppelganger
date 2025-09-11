import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional

class Parameter:
    """Replaces luigi.Parameter"""
    def __init__(self, default=None):
        self.default = default

class IntParameter(Parameter):
    """Replaces luigi.IntParameter"""
    def __init__(self, default=0):
        super().__init__(default)

class LocalTarget:
    """Replaces luigi.LocalTarget"""
    def __init__(self, path: str):
        self.path = path
    
    def exists(self) -> bool:
        return os.path.exists(self.path)
    
    def open(self, mode='r'):
        """Open file with context manager support"""
        return open(self.path, mode)

class Task(ABC):
    """Replaces luigi.Task"""
    
    def __init__(self, **kwargs):
        # Set parameters as instance attributes
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # Set default values for parameters defined in class
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if isinstance(attr, Parameter):
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, attr.default)
    
    @abstractmethod
    def run(self):
        """Task execution logic"""
        pass
    
    def requires(self) -> List['Task']:
        """Dependencies of this task"""
        return []
    
    def output(self) -> Optional[LocalTarget]:
        """Output target of this task"""
        return None
    
    def complete(self) -> bool:
        """Check if task is complete"""
        output = self.output()
        if output is None:
            return False
        return output.exists()
    
    def input(self):
        """Get input from required tasks"""
        deps = self.requires()
        if len(deps) == 1:
            return deps[0].output()
        return [dep.output() for dep in deps]

class TaskRunner:
    """Replaces luigi.build functionality"""
    
    def __init__(self):
        self.completed_tasks = set()
        self.logger = logging.getLogger(__name__)
    
    def build(self, tasks: List[Task], workers: int = 1, local_scheduler: bool = True, 
              no_lock: bool = False, log_level: str = 'INFO', detailed_summary: bool = False,
              **kwargs):
        """
        Build tasks in dependency order
        
        Args:
            tasks: List of tasks to run or single task
            workers: Number of worker processes (ignored in replacement)
            local_scheduler: Use local scheduler (always True in replacement)
            no_lock: Disable locking (ignored in replacement)
            log_level: Logging level
            detailed_summary: Show detailed summary (ignored in replacement)
            **kwargs: Additional Luigi build parameters (ignored but accepted for compatibility)
        """
        # Set up logging level
        if log_level:
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            logging.basicConfig(level=numeric_level)
            self.logger.setLevel(numeric_level)
        
        # Handle additional kwargs that Luigi might accept
        accepted_kwargs = {
            'keep_alive': kwargs.get('keep_alive', False),
            'no_configure_logging': kwargs.get('no_configure_logging', False),
            'scheduler_host': kwargs.get('scheduler_host', 'localhost'),
            'scheduler_port': kwargs.get('scheduler_port', 8082),
            'record_task_history': kwargs.get('record_task_history', False),
            'parallel_scheduling': kwargs.get('parallel_scheduling', False),
            'assistant': kwargs.get('assistant', False),
            'retcode_already_running': kwargs.get('retcode_already_running', None),
            'retcode_missing_data': kwargs.get('retcode_missing_data', None),
            'retcode_task_failed': kwargs.get('retcode_task_failed', None),
            'retcode_unhandled_exception': kwargs.get('retcode_unhandled_exception', None),
            'retcode_scheduling_error': kwargs.get('retcode_scheduling_error', None),
            'retcode_not_run': kwargs.get('retcode_not_run', None)
        }
        
        # Log configuration
        self.logger.info(f"Luigi replacement build starting with {workers} workers (ignored)")
        self.logger.info(f"Local scheduler: {local_scheduler} (always True in replacement)")
        self.logger.info(f"No lock: {no_lock} (ignored in replacement)")
        
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        success_count = 0
        total_tasks = len(tasks)
        
        try:
            for task in tasks:
                self._run_task(task)
                success_count += 1
            
            # Summary
            if detailed_summary:
                self.logger.info(f"Build summary: {success_count}/{total_tasks} tasks completed successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Build failed: {str(e)}")
            if detailed_summary:
                self.logger.info(f"Build summary: {success_count}/{total_tasks} tasks completed before failure")
            return False
    
    def _run_task(self, task: Task):
        """Run a single task and its dependencies"""
        task_id = self._get_task_id(task)
        
        if task_id in self.completed_tasks:
            return
        
        # Run dependencies first
        for dep in task.requires():
            self._run_task(dep)
        
        # Check if task is already complete
        if task.complete():
            self.logger.info(f"Task {task.__class__.__name__} already complete")
            self.completed_tasks.add(task_id)
            return
        
        # Create output directory if needed
        output = task.output()
        if output and hasattr(output, 'path'):
            output_dir = os.path.dirname(output.path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        # Run the task
        self.logger.info(f"Running task {task.__class__.__name__}")
        try:
            task.run()
            self.completed_tasks.add(task_id)
            self.logger.info(f"Task {task.__class__.__name__} completed successfully")
        except Exception as e:
            self.logger.error(f"Task {task.__class__.__name__} failed: {str(e)}")
            raise
    
    def _get_task_id(self, task: Task) -> str:
        """Generate unique task ID"""
        params = []
        for attr_name in dir(task):
            if not attr_name.startswith('_') and not callable(getattr(task, attr_name)):
                attr = getattr(task.__class__, attr_name, None)
                if isinstance(attr, Parameter):
                    params.append(f"{attr_name}={getattr(task, attr_name)}")
        
        return f"{task.__class__.__name__}({','.join(params)})"

# Global instance for compatibility
_task_runner = TaskRunner()

def build(tasks: List[Task], workers: int = 1, local_scheduler: bool = True, 
          no_lock: bool = False, log_level: str = 'INFO', detailed_summary: bool = False,
          **kwargs):
    """
    Global build function to replicate luigi.build
    
    Accepts all the same parameters as luigi.build for compatibility,
    though some are ignored in this replacement implementation.
    """
    return _task_runner.build(tasks, workers, local_scheduler, no_lock, log_level, 
                             detailed_summary, **kwargs)