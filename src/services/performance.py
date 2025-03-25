"""
Performance tracking service for SchultzGPT.
Handles timing, metrics collection, and performance reporting.
"""

import time
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import json
import statistics
from datetime import datetime, timedelta
import os


class PerformanceTracker:
    """Tracks execution time and other performance metrics."""
    
    def __init__(self, enable_file_logging: bool = False, log_file: str = "performance.log"):
        """
        Initialize the performance tracker.
        
        Args:
            enable_file_logging: Whether to log metrics to a file
            log_file: File to log metrics to
        """
        self.timers: Dict[str, List[float]] = {}
        self.success_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.token_counts: Dict[str, int] = {}
        self.custom_metrics: Dict[str, List[Any]] = {}
        self.active_timers: Dict[str, float] = {}
        self.enable_file_logging = enable_file_logging
        self.log_file = log_file
    
    def start_timer(self, name: str) -> None:
        """
        Start a timer with the given name.
        
        Args:
            name: Name of the timer
        """
        self.active_timers[name] = time.time()
    
    def stop_timer(self, name: str, success: bool = True, tokens: Optional[int] = None) -> float:
        """
        Stop a timer and record the result.
        
        Args:
            name: Name of the timer
            success: Whether the operation was successful
            tokens: Number of tokens used (for API calls)
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.active_timers:
            return 0.0
            
        elapsed = time.time() - self.active_timers[name]
        
        # Record elapsed time
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(elapsed)
        
        # Record success/error
        if success:
            self.success_counts[name] = self.success_counts.get(name, 0) + 1
        else:
            self.error_counts[name] = self.error_counts.get(name, 0) + 1
            
        # Record tokens if provided
        if tokens is not None:
            self.token_counts[name] = self.token_counts.get(name, 0) + tokens
            
        # Log to file if enabled
        if self.enable_file_logging:
            self._log_to_file(name, elapsed, success, tokens)
            
        # Remove active timer
        del self.active_timers[name]
        
        return elapsed
    
    def _log_to_file(self, name: str, elapsed: float, success: bool, tokens: Optional[int]) -> None:
        """Log performance data to a file."""
        try:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "operation": name,
                    "elapsed_seconds": elapsed,
                    "success": success
                }
                
                if tokens is not None:
                    log_entry["tokens"] = tokens
                    
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging performance data: {str(e)}")
    
    def record_metric(self, name: str, value: Any) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Name of the metric
            value: Value to record
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)
    
    def get_report(self, include_raw_data: bool = False) -> Dict[str, Any]:
        """
        Get a performance report.
        
        Args:
            include_raw_data: Whether to include raw timer data
            
        Returns:
            Performance report as a dictionary
        """
        report = {"operations": {}}
        
        for name, times in self.timers.items():
            if not times:
                continue
                
            # Calculate statistics
            avg = statistics.mean(times)
            
            # Only calculate additional stats if we have enough data
            if len(times) > 1:
                median = statistics.median(times)
                try:
                    stdev = statistics.stdev(times)
                except statistics.StatisticsError:
                    stdev = 0.0
                min_val = min(times)
                max_val = max(times)
            else:
                median = avg
                stdev = 0.0
                min_val = avg
                max_val = avg
                
            # Calculate success rate
            success_count = self.success_counts.get(name, 0)
            error_count = self.error_counts.get(name, 0)
            total_count = success_count + error_count
            
            if total_count > 0:
                success_rate = (success_count / total_count) * 100.0
            else:
                success_rate = 0.0
                
            # Build operation report
            operation_report = {
                "avg_seconds": avg,
                "median_seconds": median,
                "stdev_seconds": stdev,
                "min_seconds": min_val,
                "max_seconds": max_val,
                "calls": len(times),
                "success_rate": success_rate,
                "tokens": self.token_counts.get(name, 0)
            }
            
            # Add raw data if requested
            if include_raw_data:
                operation_report["raw_times"] = times
                
            report["operations"][name] = operation_report
            
        # Add custom metrics
        report["custom_metrics"] = {}
        for name, values in self.custom_metrics.items():
            if not values:
                continue
                
            # Calculate basic statistics if values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                avg = statistics.mean(values)
                if len(values) > 1:
                    median = statistics.median(values)
                    try:
                        stdev = statistics.stdev(values)
                    except statistics.StatisticsError:
                        stdev = 0.0
                    min_val = min(values)
                    max_val = max(values)
                else:
                    median = avg
                    stdev = 0.0
                    min_val = avg
                    max_val = avg
                    
                report["custom_metrics"][name] = {
                    "avg": avg,
                    "median": median,
                    "stdev": stdev,
                    "min": min_val,
                    "max": max_val,
                    "count": len(values)
                }
            else:
                # Just include count for non-numeric values
                report["custom_metrics"][name] = {
                    "count": len(values)
                }
                
        return report
    
    def get_formatted_report(self) -> str:
        """
        Get a formatted performance report as a string.
        
        Returns:
            Formatted report
        """
        report = self.get_report()
        formatted_report = "Performance Report:\n"
        
        # Add operations
        formatted_report += "\nOperations:\n"
        for name, stats in report["operations"].items():
            formatted_report += f"  {name}:\n"
            formatted_report += f"    Avg: {stats['avg_seconds']:.4f}s\n"
            formatted_report += f"    Median: {stats['median_seconds']:.4f}s\n"
            formatted_report += f"    Min/Max: {stats['min_seconds']:.4f}s / {stats['max_seconds']:.4f}s\n"
            formatted_report += f"    Calls: {stats['calls']}\n"
            formatted_report += f"    Success Rate: {stats['success_rate']:.1f}%\n"
            
            if stats['tokens'] > 0:
                formatted_report += f"    Tokens: {stats['tokens']}\n"
                
        # Add custom metrics
        if report["custom_metrics"]:
            formatted_report += "\nCustom Metrics:\n"
            for name, stats in report["custom_metrics"].items():
                formatted_report += f"  {name}:\n"
                if "avg" in stats:
                    formatted_report += f"    Avg: {stats['avg']:.4f}\n"
                    formatted_report += f"    Median: {stats['median']:.4f}\n"
                    formatted_report += f"    Min/Max: {stats['min']:.4f} / {stats['max']:.4f}\n"
                
                formatted_report += f"    Count: {stats['count']}\n"
                
        return formatted_report
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.timers = {}
        self.success_counts = {}
        self.error_counts = {}
        self.token_counts = {}
        self.custom_metrics = {}
        self.active_timers = {}


def timed(func=None, *, name=None, tracker=None):
    """
    Decorator for timing function execution.
    
    Can be used either as:
        @timed
        def my_function():
            # ... (requires self.performance_tracker in class)
            
        @timed(name="custom_name")
        def my_function():
            # ... (requires self.performance_tracker in class)
            
        @timed(tracker=my_tracker)
        def my_function():
            # ...
            
        @timed(name="custom_name", tracker=my_tracker)
        def my_function():
            # ...
    """
    if func is None:
        # Called with parameters - @timed(name="something")
        def decorator_with_args(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # Determine which tracker to use
                nonlocal tracker
                if tracker is not None:
                    perf_tracker = tracker
                else:
                    # Try to get tracker from self if this is a method
                    if args and hasattr(args[0], "performance_tracker"):
                        perf_tracker = args[0].performance_tracker
                    else:
                        raise ValueError("No performance tracker available. Pass a tracker explicitly or ensure self.performance_tracker exists.")
                
                # Determine timer name
                nonlocal name
                timer_name = name or fn.__name__
                
                # Start timer and execute function
                perf_tracker.start_timer(timer_name)
                try:
                    result = fn(*args, **kwargs)
                    perf_tracker.stop_timer(timer_name, success=True)
                    return result
                except Exception as e:
                    perf_tracker.stop_timer(timer_name, success=False)
                    raise e
            return wrapper
        return decorator_with_args
    else:
        # Called without parameters - @timed
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get tracker from self if this is a method
            if args and hasattr(args[0], "performance_tracker"):
                perf_tracker = args[0].performance_tracker
            else:
                raise ValueError("No performance tracker available. Pass a tracker explicitly or ensure self.performance_tracker exists.")
            
            # Start timer and execute function
            timer_name = func.__name__
            perf_tracker.start_timer(timer_name)
            try:
                result = func(*args, **kwargs)
                perf_tracker.stop_timer(timer_name, success=True)
                return result
            except Exception as e:
                perf_tracker.stop_timer(timer_name, success=False)
                raise e
        return wrapper


async def async_timed(func=None, *, name=None, tracker=None):
    """
    Decorator for timing async function execution.
    
    Can be used either as:
        @async_timed
        async def my_function():
            # ... (requires self.performance_tracker in class)
            
        @async_timed(name="custom_name")
        async def my_function():
            # ... (requires self.performance_tracker in class)
            
        @async_timed(tracker=my_tracker)
        async def my_function():
            # ...
            
        @async_timed(name="custom_name", tracker=my_tracker)
        async def my_function():
            # ...
    """
    if func is None:
        # Called with parameters - @async_timed(name="something")
        def decorator_with_args(fn):
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                # Determine which tracker to use
                nonlocal tracker
                if tracker is not None:
                    perf_tracker = tracker
                else:
                    # Try to get tracker from self if this is a method
                    if args and hasattr(args[0], "performance_tracker"):
                        perf_tracker = args[0].performance_tracker
                    else:
                        raise ValueError("No performance tracker available. Pass a tracker explicitly or ensure self.performance_tracker exists.")
                
                # Determine timer name
                nonlocal name
                timer_name = name or fn.__name__
                
                # Start timer and execute function
                perf_tracker.start_timer(timer_name)
                try:
                    result = await fn(*args, **kwargs)
                    perf_tracker.stop_timer(timer_name, success=True)
                    return result
                except Exception as e:
                    perf_tracker.stop_timer(timer_name, success=False)
                    raise e
            return wrapper
        return decorator_with_args
    else:
        # Called without parameters - @async_timed
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to get tracker from self if this is a method
            if args and hasattr(args[0], "performance_tracker"):
                perf_tracker = args[0].performance_tracker
            else:
                raise ValueError("No performance tracker available. Pass a tracker explicitly or ensure self.performance_tracker exists.")
            
            # Start timer and execute function
            timer_name = func.__name__
            perf_tracker.start_timer(timer_name)
            try:
                result = await func(*args, **kwargs)
                perf_tracker.stop_timer(timer_name, success=True)
                return result
            except Exception as e:
                perf_tracker.stop_timer(timer_name, success=False)
                raise e
        return wrapper 