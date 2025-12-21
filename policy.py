import numpy as np

class TanhMLP:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Calculate shapes
        self.w1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.w2_size = hidden_dim * output_dim
        self.b2_size = output_dim
        
        self.total_weights = (self.w1_size + self.b1_size + 
                              self.w2_size + self.b2_size)

    def unpack(self, genome):
        """Slice the flat genome vector into weight matrices."""
        idx = 0
        w1 = genome[idx : idx+self.w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += self.w1_size
        
        b1 = genome[idx : idx+self.b1_size]
        idx += self.b1_size
        
        w2 = genome[idx : idx+self.w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += self.b2_size # Bug fix: increment index correctly
        
        b2 = genome[idx : idx+self.b2_size]
        
        return w1, b1, w2, b2

    def forward(self, state, genome):
        w1, b1, w2, b2 = self.unpack(genome)
        
        # Layer 1
        x = np.tanh(state @ w1 + b1)
        # Layer 2 (Output)
        action = np.tanh(x @ w2 + b2)
        
        return action
