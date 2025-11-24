import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

input_data = np.random.randn(1, 50).astype(np.float32)

inputs = httpclient.InferInput("INPUT__0", input_data.shape, "FP32")
inputs.set_data_from_numpy(input_data)

outputs = httpclient.InferRequestedOutput("OUTPUT__0")

response = client.infer("lit-autoencoder", inputs=[inputs], outputs=[outputs])
print(response.as_numpy("OUTPUT__0"))
