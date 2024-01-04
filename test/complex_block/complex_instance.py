import torch
import torch.nn as nn


class ComplexInstanceNormalization(nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(ComplexInstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, input):
        # Calculate mean and variance for each instance (sample)
        mean = input.mean(dim=(2, 3), keepdim=True)
        var = input.var(dim=(2, 3), unbiased=False, keepdim=True)

        # Normalize
        input_normalized = (input - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        output = self.gamma * input_normalized + self.beta

        return output


# Example usage:
# Assume input_data is a complex-valued input tensor
input_data = torch.randn(10, 2, 32, 32) + 1j * torch.randn(10, 2, 32, 32)
instance_norm_layer = ComplexInstanceNormalization(num_features=2)
output = instance_norm_layer(input_data)
print(input_data)
print(output.size())
print(input_data.size())

# The output tensor 'output' is the result of applying complex instance normalization
