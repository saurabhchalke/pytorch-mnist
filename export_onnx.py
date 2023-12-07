import torch
from model import Net  # Import your model definition

# Load the trained model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))  # Lzoad your model weights
model.eval()

# Create dummy input in the shape the model expects
dummy_input = torch.randn(1, 1, 28, 28)  # Example for MNIST

# Export the model
torch.onnx.export(model, dummy_input, "mnist_model.onnx", verbose=True)

print("Model exported to mnist_model.onnx")
