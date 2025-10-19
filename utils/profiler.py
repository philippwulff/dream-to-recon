import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T


class Profiler:
    """
    https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    
    Installing the perfetto trace processor locally to view large traces:
    
    curl -LO https://get.perfetto.dev/trace_processor
    chmod +x ./trace_processor
    
    Running the trace processor:
    
    trace_processor --httpd /path/to/trace.pftrace

    """
    def __init__(self, tb_log_path, wait=1, warmup=1, active=3, repeat=1) -> None:
        
        self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait,      # Forwards without measuring
                    warmup=warmup,  # Measurements thrown away
                    active=active,  # Active recording
                    repeat=repeat   # Repeat cycle starting with the wait steps
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_path),
                record_shapes=False, 
                profile_memory=False,
                with_stack=True,         # Records stack trace of functions -> causes large trace files
                with_modules=False,
            )
        # Called once before model loop.
        self.prof.start()
        
    def step(self):
        """Called before every iteration."""
        self.prof.step()
        
    def cleanup(self):
        """Called after the last iteration."""
        self.prof.stop()
