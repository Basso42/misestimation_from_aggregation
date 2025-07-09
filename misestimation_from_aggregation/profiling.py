"""
Performance profiling and benchmarking tools for the misestimation_from_aggregation package.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from functools import wraps
import sys
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PerformanceProfiler:
    """
    A performance profiler for tracking execution times and memory usage.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking and HAS_PSUTIL
        self.profiles = {}
        self.current_profile = None
        
    def start_profiling(self, name: str) -> None:
        """Start profiling a section of code."""
        if self.current_profile is not None:
            warnings.warn(f"Starting new profile '{name}' while '{self.current_profile}' is still active")
        
        self.current_profile = name
        self.profiles[name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'end_time': None,
            'end_memory': None,
            'duration': None,
            'memory_delta': None
        }
    
    def end_profiling(self, name: Optional[str] = None) -> Dict:
        """End profiling and return results."""
        if name is None:
            name = self.current_profile
        
        if name is None or name not in self.profiles:
            raise ValueError(f"No active profile named '{name}'")
        
        profile = self.profiles[name]
        profile['end_time'] = time.time()
        profile['end_memory'] = self._get_memory_usage()
        profile['duration'] = profile['end_time'] - profile['start_time']
        
        if profile['start_memory'] is not None and profile['end_memory'] is not None:
            profile['memory_delta'] = profile['end_memory'] - profile['start_memory']
        
        if self.current_profile == name:
            self.current_profile = None
        
        return profile
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking:
            return None
        
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return None
    
    def get_summary(self) -> Dict:
        """Get summary of all profiles."""
        summary = {}
        for name, profile in self.profiles.items():
            if profile['duration'] is not None:
                summary[name] = {
                    'duration': profile['duration'],
                    'memory_delta': profile['memory_delta']
                }
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of all profiles."""
        print("\nPerformance Profile Summary")
        print("=" * 50)
        
        for name, profile in self.profiles.items():
            if profile['duration'] is not None:
                print(f"{name}:")
                print(f"  Duration: {profile['duration']:.4f}s")
                if profile['memory_delta'] is not None:
                    print(f"  Memory: {profile['memory_delta']:+.1f}MB")
                print()


def profile_function(profiler: PerformanceProfiler, name: Optional[str] = None):
    """
    Decorator to profile function execution.
    
    Parameters
    ----------
    profiler : PerformanceProfiler
        The profiler instance to use
    name : str, optional
        Name for the profile (defaults to function name)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile_name = name or func.__name__
            
            profiler.start_profiling(profile_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_profiling(profile_name)
        
        return wrapper
    return decorator


class BenchmarkSuite:
    """
    A comprehensive benchmark suite for the package.
    """
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
    
    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> Dict:
        """
        Run a benchmark for a specific function.
        
        Parameters
        ----------
        name : str
            Name of the benchmark
        func : Callable
            Function to benchmark
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        dict
            Benchmark results
        """
        print(f"Running benchmark: {name}")
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        benchmark_result = {
            'name': name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'times': times.tolist()
        }
        
        self.results[name] = benchmark_result
        print(f"  Mean: {benchmark_result['mean_time']:.4f}s ± {benchmark_result['std_time']:.4f}s")
        
        return benchmark_result
    
    def compare_implementations(self, name1: str, name2: str) -> None:
        """
        Compare two benchmark implementations.
        
        Parameters
        ----------
        name1 : str
            Name of first implementation
        name2 : str
            Name of second implementation
        """
        if name1 not in self.results or name2 not in self.results:
            raise ValueError("Both implementations must be benchmarked first")
        
        result1 = self.results[name1]
        result2 = self.results[name2]
        
        speedup = result1['mean_time'] / result2['mean_time']
        
        print(f"\nComparison: {name1} vs {name2}")
        print(f"  {name1}: {result1['mean_time']:.4f}s ± {result1['std_time']:.4f}s")
        print(f"  {name2}: {result2['mean_time']:.4f}s ± {result2['std_time']:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"  {name2} is {speedup:.2f}x faster than {name1}")
        else:
            print(f"  {name1} is {1/speedup:.2f}x faster than {name2}")
    
    def print_summary(self) -> None:
        """Print a summary of all benchmarks."""
        print("\nBenchmark Summary")
        print("=" * 50)
        
        for name, result in self.results.items():
            print(f"{name}:")
            print(f"  Mean: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
            print(f"  Range: {result['min_time']:.4f}s - {result['max_time']:.4f}s")
            print()


def create_performance_report(package_name: str = "misestimation_from_aggregation") -> str:
    """
    Create a comprehensive performance report.
    
    Parameters
    ----------
    package_name : str
        Name of the package
        
    Returns
    -------
    str
        Formatted performance report
    """
    report = []
    report.append(f"Performance Report for {package_name}")
    report.append("=" * 60)
    
    # System info
    report.append("\nSystem Information:")
    report.append(f"  Python version: {sys.version}")
    report.append(f"  NumPy version: {np.__version__}")
    
    if HAS_PSUTIL:
        report.append(f"  CPU cores: {psutil.cpu_count()}")
        report.append(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # Package info
    try:
        import misestimation_from_aggregation
        report.append(f"  Package version: {misestimation_from_aggregation.__version__}")
    except:
        report.append("  Package version: unknown")
    
    return "\n".join(report)


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile(name: Optional[str] = None):
    """
    Decorator to profile function execution using the global profiler.
    
    Parameters
    ----------
    name : str, optional
        Name for the profile (defaults to function name)
    """
    return profile_function(_global_profiler, name)