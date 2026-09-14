import torch
import torch.nn as nn

class LoopedStack(nn.Module):
    def __init__(self, blocks, n_loops):
        super().__init__()
        # Store the blocks in an nn.ModuleList so PyTorch registers their parameters
        self.blocks = nn.ModuleList(blocks) if not isinstance(blocks, nn.ModuleList) else blocks
        self.n_loops = n_loops

    def forward(self, x, n_loops=None):
        # Use provided n_loops or fall back to the initialized default
        loops = n_loops if n_loops is not None else self.n_loops
        
        # Apply every block in order, n_loops times, reusing the same blocks
        for _ in range(loops):
            for block in self.blocks:
                x = block(x)
                
        return x
        
    def block_applications(self, n_loops=None):
        # Number of block applications is the number of blocks times the number of loops
        loops = n_loops if n_loops is not None else self.n_loops
        return len(self.blocks) * loops

    def shared_parameter_count(self):
        # Counts stored parameters exactly once since weights are shared across loops
        return sum(p.numel() for p in self.parameters())

    def unrolled_parameter_count(self, n_loops=None):
        # Unrolled count is the stored count times the number of loops
        loops = n_loops if n_loops is not None else self.n_loops
        return self.shared_parameter_count() * loops