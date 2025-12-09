import torch

# Assuming you have imported the required classes and modules from the provided code
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from layers import CheckerboardMaskedConv2d
# from modules import Demultiplexer, Multiplexer

class CheckerboardAutogressive(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=4, M=32, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardMaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

    def forward(self, y):
        # Process input tensor y through the model
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        # Scale the input tensor y by 10, quantize it, then divide by 10
        y_hat = self.gaussian_conditional.quantize(y.mul_(10), "noise")
        y_hat.div_(10)

        # Set non-anchor points in y_hat to 0
        y_half = y_hat.clone()
        y_half[:, :, 0::2, 0::2] = 0
        y_half[:, :, 1::2, 1::2] = 0

        # Use context prediction on the non-anchor points
        ctx_params = self.context_prediction(y_half)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0

        # Combine the parameters and context prediction for the Gaussian conditional model
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # Process the quantized y_hat and compute the likelihoods
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        # Output the shape of y_likelihoods
        print("Shape of y_likelihoods:", y_likelihoods.shape)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

# Test the model with a 1x32x64x64 tensor
if __name__ == "__main__":
    model = CheckerboardAutogressive()
    
    # Create a dummy input tensor of size (1, 32, 64, 64)
    input_tensor = torch.randn(1, 32, 64, 64)
    
    # Run the forward pass through the model
    output = model(input_tensor)
    
    # Check the shape of the output
    print("Output shapes:")
    print("x_hat:", output["x_hat"].shape)
    print("y likelihoods:", output["likelihoods"]["y"].shape)
    print("z likelihoods:", output["likelihoods"]["z"].shape)
