import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("mnist_model.onnx")

# Prepare dummy input data as a numpy array (example input)
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)  # Example input

# Run the model (forward pass)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: input_data})

print("ONNX model output:", output)
