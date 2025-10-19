from dataclasses import dataclass


@dataclass
class ProfilerConfig:
    ENABLED: bool = False
    TRACE_SUBDIR_NAME: str = "profiling_traces"
    # The steps in the profiling cycle
    WAIT: int = 1
    WARMUP: int = 1
    ACTIVE: int = 3
    REPEAT: int = 1
    
    def get_total_profiling_steps(self):
        return self.REPEAT * (self.WAIT + self.WARMUP + self.ACTIVE)