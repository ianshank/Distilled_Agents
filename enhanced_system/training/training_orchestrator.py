"""
Training Orchestration Module
Manages parallel training of multiple agents with dependency tracking
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job configuration"""
    job_id: str
    agent_name: str
    dataset_path: str
    model_config: Dict[str, Any]
    dependencies: List[str]  # IDs of jobs that must complete first
    status: TrainingStatus = TrainingStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class TrainingOrchestrator:
    """
    Orchestrate multi-agent training with dependencies
    
    Features:
    - Dependency graph management
    - Parallel training execution
    - Progress tracking
    - Resource optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training orchestrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.parallel_training = config.get('parallel_training', True)
        self.max_parallel_jobs = config.get('max_parallel_jobs', 3)
        self.dependency_tracking = config.get('dependency_tracking', True)
        
        # Resource limits
        resource_limits = config.get('resource_limits', {})
        self.max_memory_gb = resource_limits.get('max_memory_gb', 32)
        self.max_gpus = resource_limits.get('max_gpus', 2)
        
        # Job tracking
        self.jobs: Dict[str, TrainingJob] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.running_jobs: Set[str] = set()
        
        logger.info(
            f"TrainingOrchestrator initialized (max_parallel: {self.max_parallel_jobs})"
        )
    
    def add_training_job(
        self,
        job_id: str,
        agent_name: str,
        dataset_path: str,
        model_config: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Add training job to orchestrator
        
        Args:
            job_id: Unique job identifier
            agent_name: Name of agent to train
            dataset_path: Path to training dataset
            model_config: Model configuration
            dependencies: Optional list of dependent job IDs
        
        Returns:
            Job ID
        """
        job = TrainingJob(
            job_id=job_id,
            agent_name=agent_name,
            dataset_path=dataset_path,
            model_config=model_config,
            dependencies=dependencies or []
        )
        
        self.jobs[job_id] = job
        
        # Build dependency graph
        if self.dependency_tracking:
            self.dependency_graph[job_id] = set(dependencies or [])
        
        logger.info(f"Added training job: {job_id} (agent: {agent_name})")
        
        return job_id
    
    async def train_all(self) -> Dict[str, TrainingJob]:
        """
        Train all agents respecting dependencies
        
        Returns:
            Dictionary of job results
        """
        if not self.jobs:
            logger.warning("No training jobs to execute")
            return {}
        
        logger.info(f"Starting training for {len(self.jobs)} jobs")
        
        completed_jobs: Set[str] = set()
        failed_jobs: Set[str] = set()
        
        while len(completed_jobs) + len(failed_jobs) < len(self.jobs):
            # Find jobs ready to run
            ready_jobs = self._get_ready_jobs(completed_jobs, failed_jobs)
            
            if not ready_jobs:
                # Check if we're stuck
                if not self.running_jobs:
                    logger.error("No jobs ready and none running - circular dependency?")
                    break
                # Wait for running jobs to complete
                await asyncio.sleep(1)
                continue
            
            # Limit parallel jobs
            available_slots = self.max_parallel_jobs - len(self.running_jobs)
            jobs_to_start = ready_jobs[:available_slots]
            
            if not jobs_to_start:
                # All slots full, wait
                await asyncio.sleep(1)
                continue
            
            # Start jobs
            tasks = []
            for job_id in jobs_to_start:
                self.running_jobs.add(job_id)
                task = asyncio.create_task(self._train_agent(job_id))
                tasks.append((job_id, task))
            
            # Wait for at least one to complete
            if tasks:
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    # Find corresponding job_id
                    for job_id, job_task in tasks:
                        if job_task == task:
                            self.running_jobs.discard(job_id)
                            
                            try:
                                result = await task
                                if result['success']:
                                    completed_jobs.add(job_id)
                                    logger.info(f"Job {job_id} completed successfully")
                                else:
                                    failed_jobs.add(job_id)
                                    logger.error(f"Job {job_id} failed")
                            except Exception as e:
                                failed_jobs.add(job_id)
                                logger.error(f"Job {job_id} failed with exception: {e}")
                            
                            break
        
        logger.info(
            f"Training complete: {len(completed_jobs)} succeeded, "
            f"{len(failed_jobs)} failed"
        )
        
        return self.jobs
    
    def _get_ready_jobs(
        self,
        completed: Set[str],
        failed: Set[str]
    ) -> List[str]:
        """
        Get jobs that are ready to run
        
        Args:
            completed: Set of completed job IDs
            failed: Set of failed job IDs
        
        Returns:
            List of ready job IDs
        """
        ready = []
        
        for job_id, job in self.jobs.items():
            # Skip if already processed or running
            if (job_id in completed or 
                job_id in failed or 
                job_id in self.running_jobs):
                continue
            
            # Check dependencies
            if self.dependency_tracking:
                deps = self.dependency_graph.get(job_id, set())
                
                # Check if any dependency failed
                if any(dep in failed for dep in deps):
                    # Mark as failed due to dependency failure
                    self.jobs[job_id].status = TrainingStatus.FAILED
                    self.jobs[job_id].error_message = "Dependency failed"
                    failed.add(job_id)
                    continue
                
                # Check if all dependencies completed
                if not all(dep in completed for dep in deps):
                    continue
            
            ready.append(job_id)
        
        return ready
    
    async def _train_agent(self, job_id: str) -> Dict[str, Any]:
        """
        Train a single agent
        
        Args:
            job_id: Job identifier
        
        Returns:
            Training result
        """
        job = self.jobs[job_id]
        
        logger.info(f"Starting training job {job_id} (agent: {job.agent_name})")
        
        job.status = TrainingStatus.RUNNING
        job.start_time = datetime.now()
        
        try:
            # Simulate training (in real implementation, call actual training)
            result = await self._execute_training(job)
            
            job.status = TrainingStatus.COMPLETED
            job.end_time = datetime.now()
            job.metrics = result.get('metrics', {})
            
            logger.info(
                f"Job {job_id} completed in "
                f"{(job.end_time - job.start_time).total_seconds():.1f}s"
            )
            
            return {'success': True, 'metrics': job.metrics}
        
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            
            logger.error(f"Job {job_id} failed: {e}")
            
            return {'success': False, 'error': str(e)}
    
    async def _execute_training(self, job: TrainingJob) -> Dict[str, Any]:
        """
        Execute actual training
        
        Args:
            job: Training job
        
        Returns:
            Training results
        """
        # Simulate training process
        logger.info(f"Training {job.agent_name} with dataset {job.dataset_path}")
        
        # Simulate training time
        await asyncio.sleep(2)
        
        # Return mock metrics
        return {
            'metrics': {
                'train_loss': 0.45,
                'val_loss': 0.52,
                'accuracy': 0.87,
                'epochs': job.model_config.get('epochs', 5),
                'training_time_s': 120
            }
        }
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, TrainingJob]:
        """Get all training jobs"""
        return self.jobs.copy()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress"""
        total = len(self.jobs)
        completed = sum(1 for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED)
        failed = sum(1 for j in self.jobs.values() if j.status == TrainingStatus.FAILED)
        running = len(self.running_jobs)
        pending = total - completed - failed - running
        
        return {
            'total_jobs': total,
            'completed': completed,
            'failed': failed,
            'running': running,
            'pending': pending,
            'progress_pct': (completed / total * 100) if total > 0 else 0
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a training job
        
        Args:
            job_id: Job ID to cancel
        
        Returns:
            True if cancelled
        """
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
                job.status = TrainingStatus.CANCELLED
                self.running_jobs.discard(job_id)
                logger.info(f"Cancelled job {job_id}")
                return True
        
        return False

