import onnx
import onnxruntime as ort

# Load ONNX model
model_path = "yash.onnx"
model = onnx.load(model_path)
onnx.checker.check_model(model)

# Show input and output names
print("Inputs:")
for input_tensor in model.graph.input:
    print(f"  Name: {input_tensor.name}")

print("\nOutputs:")
for output_tensor in model.graph.output:
    print(f"  Name: {output_tensor.name}")

# Optional: Get metadata (sometimes contains class names)
print("\nModel Metadata:")
for prop in model.metadata_props:
    print(f"  {prop.key}: {prop.value}")
