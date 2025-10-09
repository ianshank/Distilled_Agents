"""
Error Handling and Fallback Management Module
Provides intelligent retry logic, fallback strategies, and error learning
"""

import logging
import asyncio
import sqlite3
import json
from typing import Callable, Any, Optional, List, Dict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types"""
    TEMPORARY = "temporary"  # Retryable errors (network, timeout, rate limit)
    PERMANENT = "permanent"  # Non-retryable errors (invalid input, auth failure)
    UNKNOWN = "unknown"      # Unclassified errors


class TemporaryError(Exception):
    """Exception for temporary/retryable errors"""
    pass


class PermanentError(Exception):
    """Exception for permanent/non-retryable errors"""
    pass


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_type: str
    error_message: str
    agent: str
    task: str
    timestamp: datetime
    retry_count: int
    resolved: bool
    resolution_method: Optional[str] = None


class ErrorLearner:
    """
    Learn from errors to improve system
    
    Tracks error patterns and generates recommendations
    """
    
    def __init__(self, db_path: str = "./data/errors.db"):
        """
        Initialize error learner
        
        Args:
            db_path: Path to error database
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize error database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT,
                    error_message TEXT,
                    agent TEXT,
                    task TEXT,
                    timestamp TEXT,
                    retry_count INTEGER,
                    resolved INTEGER,
                    resolution_method TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_error_type ON errors(error_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent ON errors(agent)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Error database initialized at {self.db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize error database: {e}")
    
    def record_error(self, error_record: ErrorRecord) -> None:
        """Record an error occurrence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO errors (
                    error_type, error_message, agent, task, timestamp,
                    retry_count, resolved, resolution_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error_record.error_type,
                error_record.error_message,
                error_record.agent,
                error_record.task,
                error_record.timestamp.isoformat(),
                error_record.retry_count,
                1 if error_record.resolved else 0,
                error_record.resolution_method
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Error recorded: {error_record.error_type}")
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def analyze_errors(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze error patterns
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Analysis results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get errors from last N days
            cursor.execute('''
                SELECT error_type, error_message, agent, COUNT(*) as count
                FROM errors
                WHERE timestamp >= datetime('now', ? || ' days')
                GROUP BY error_type, agent
                ORDER BY count DESC
                LIMIT 10
            ''', (f"-{days}",))
            
            error_patterns = [
                {
                    'error_type': row[0],
                    'error_message': row[1],
                    'agent': row[2],
                    'count': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Get resolution success rate
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN resolved = 1 THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM errors
                WHERE timestamp >= datetime('now', ? || ' days')
            ''', (f"-{days}",))
            
            success_rate = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'error_patterns': error_patterns,
                'success_rate': success_rate,
                'analysis_period_days': days
            }
        except Exception as e:
            logger.error(f"Failed to analyze errors: {e}")
            return {
                'error_patterns': [],
                'success_rate': 0,
                'analysis_period_days': days
            }


class IntelligentRetryHandler:
    """
    Intelligent retry logic with exponential backoff
    
    Features:
    - Exponential backoff
    - Context-aware retry decisions
    - Error classification
    - Parameter adjustment between retries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize retry handler
        
        Args:
            config: Retry configuration
        """
        self.max_retries = config.get('max_retries', 3)
        self.base_delay = config.get('base_delay', 1.0)
        self.max_delay = config.get('max_delay', 32.0)
        self.exponential_base = config.get('exponential_base', 2)
        self.error_learner = None
        
        if config.get('enable_error_learning', True):
            try:
                self.error_learner = ErrorLearner(
                    config.get('error_db_path', './data/errors.db')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize error learner: {e}")
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify error as temporary or permanent
        
        Args:
            error: Exception to classify
        
        Returns:
            ErrorType classification
        """
        if isinstance(error, TemporaryError):
            return ErrorType.TEMPORARY
        
        if isinstance(error, PermanentError):
            return ErrorType.PERMANENT
        
        # Classify based on error message
        error_str = str(error).lower()
        
        # Temporary error indicators
        temporary_indicators = [
            'timeout', 'connection', 'network', 'rate limit',
            'throttle', 'busy', 'unavailable', 'retry'
        ]
        
        for indicator in temporary_indicators:
            if indicator in error_str:
                return ErrorType.TEMPORARY
        
        # Permanent error indicators
        permanent_indicators = [
            'invalid', 'forbidden', 'unauthorized', 'not found',
            'permission', 'authentication', 'malformed'
        ]
        
        for indicator in permanent_indicators:
            return ErrorType.PERMANENT
        
        return ErrorType.UNKNOWN
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt
        
        Args:
            attempt: Retry attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        agent: str = "unknown",
        task: str = "",
        **kwargs
    ) -> Any:
        """
        Execute function with intelligent retry
        
        Args:
            func: Function to execute
            *args: Function arguments
            agent: Agent identifier for error tracking
            task: Task description
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception if all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - record if we had retries
                if attempt > 0 and self.error_learner:
                    self.error_learner.record_error(ErrorRecord(
                        error_type="resolved",
                        error_message=str(last_error),
                        agent=agent,
                        task=task,
                        timestamp=datetime.now(),
                        retry_count=attempt,
                        resolved=True,
                        resolution_method="retry"
                    ))
                
                return result
            
            except Exception as error:
                last_error = error
                error_type = self.classify_error(error)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {error} "
                    f"(type: {error_type.value})"
                )
                
                # Don't retry permanent errors
                if error_type == ErrorType.PERMANENT:
                    logger.error(f"Permanent error detected, not retrying: {error}")
                    if self.error_learner:
                        self.error_learner.record_error(ErrorRecord(
                            error_type=error_type.value,
                            error_message=str(error),
                            agent=agent,
                            task=task,
                            timestamp=datetime.now(),
                            retry_count=attempt,
                            resolved=False
                        ))
                    raise
                
                # Check if we have retries left
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    logger.error(f"All {self.max_retries} retries exhausted")
                    if self.error_learner:
                        self.error_learner.record_error(ErrorRecord(
                            error_type=error_type.value,
                            error_message=str(error),
                            agent=agent,
                            task=task,
                            timestamp=datetime.now(),
                            retry_count=attempt + 1,
                            resolved=False
                        ))
                    raise


class FallbackManager:
    """
    Multi-level fallback strategy manager
    
    Implements a chain of fallback strategies when primary execution fails
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fallback manager
        
        Args:
            config: Fallback configuration
        """
        self.config = config
        self.enable_fallback = config.get('enable_fallback', True)
        self.fallback_chain = []
        self.error_learner = None
        
        if config.get('enable_error_learning', True):
            try:
                self.error_learner = ErrorLearner(
                    config.get('error_db_path', './data/errors.db')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize error learner: {e}")
    
    def register_fallback(
        self,
        strategy: Callable,
        priority: int = 100,
        name: str = "unnamed"
    ) -> None:
        """
        Register a fallback strategy
        
        Args:
            strategy: Fallback strategy function
            priority: Priority (lower = higher priority)
            name: Strategy name
        """
        self.fallback_chain.append({
            'strategy': strategy,
            'priority': priority,
            'name': name
        })
        # Sort by priority
        self.fallback_chain.sort(key=lambda x: x['priority'])
        logger.info(f"Registered fallback strategy: {name} (priority: {priority})")
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        *args,
        agent: str = "unknown",
        task: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with fallback strategies
        
        Args:
            primary_func: Primary function to execute
            *args: Function arguments
            agent: Agent identifier
            task: Task description
            **kwargs: Function keyword arguments
        
        Returns:
            Dict with result and metadata
        """
        if not self.enable_fallback:
            # No fallback enabled, just execute primary
            result = await primary_func(*args, **kwargs) if asyncio.iscoroutinefunction(primary_func) else primary_func(*args, **kwargs)
            return {
                'result': result,
                'method': 'primary',
                'success': True
            }
        
        # Try primary function
        try:
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)
            
            return {
                'result': result,
                'method': 'primary',
                'success': True
            }
        except Exception as primary_error:
            logger.warning(f"Primary execution failed: {primary_error}")
        
        # Try fallback strategies
        for fallback in self.fallback_chain:
            strategy_name = fallback['name']
            strategy_func = fallback['strategy']
            
            try:
                logger.info(f"Trying fallback strategy: {strategy_name}")
                
                if asyncio.iscoroutinefunction(strategy_func):
                    result = await strategy_func(*args, **kwargs)
                else:
                    result = strategy_func(*args, **kwargs)
                
                # Check if result is acceptable
                if self._is_acceptable_result(result):
                    logger.info(f"Fallback strategy succeeded: {strategy_name}")
                    
                    # Record success
                    if self.error_learner:
                        self.error_learner.record_error(ErrorRecord(
                            error_type="resolved",
                            error_message=str(primary_error),
                            agent=agent,
                            task=task,
                            timestamp=datetime.now(),
                            retry_count=0,
                            resolved=True,
                            resolution_method=strategy_name
                        ))
                    
                    return {
                        'result': result,
                        'method': strategy_name,
                        'success': True,
                        'fallback': True
                    }
            
            except Exception as fallback_error:
                logger.warning(
                    f"Fallback strategy {strategy_name} failed: {fallback_error}"
                )
                continue
        
        # All strategies failed
        logger.error("All fallback strategies failed")
        if self.error_learner:
            self.error_learner.record_error(ErrorRecord(
                error_type="all_failed",
                error_message=str(primary_error),
                agent=agent,
                task=task,
                timestamp=datetime.now(),
                retry_count=len(self.fallback_chain),
                resolved=False
            ))
        
        raise Exception(f"All strategies failed. Primary error: {primary_error}")
    
    def _is_acceptable_result(self, result: Any) -> bool:
        """
        Check if fallback result is acceptable
        
        Args:
            result: Result to check
        
        Returns:
            True if acceptable
        """
        # Basic checks
        if result is None:
            return False
        
        # If result is a dict with success flag
        if isinstance(result, dict) and 'success' in result:
            return result['success']
        
        # If result is a string, check if it's not empty
        if isinstance(result, str):
            return len(result.strip()) > 0
        
        # Default: accept non-None results
        return True


class AllStrategiesFailedError(Exception):
    """Exception when all fallback strategies fail"""
    pass

