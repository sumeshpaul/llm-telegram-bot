from llama_cpp import Llama

# Set your model path
MODEL_PATH = "/mnt/mymodels/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=50,  # Adjust based on your GPU (e.g., RTX 5080)
    use_mlock=True
)

# Simple prompt-based call (like /generate)
def llm_model(prompt, max_tokens=1024, stop=None):
    response = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        stop=stop,
        echo=False
    )
    return response

# Chat-style call (like /chat)
def llm_model_create_chat_completion(messages, max_tokens=1024, temperature=0.7, stop=None):
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop
    )
    return response
