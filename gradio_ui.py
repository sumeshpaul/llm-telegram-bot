import gradio as gr
import requests
import os
import psutil

API_URL = "http://localhost:8000/predict"
API_KEY = "DESKNAV-AI-2025-SECURE"

# 🔧 Auto-kill process using port 7863
def free_port(port=7863):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.status == 'LISTEN' and conn.laddr.port == port:
                    print(f"⚠️ Killing process {proc.pid} on port {port}")
                    try:
                        proc.kill()
                    except Exception as e:
                        print(f"❌ Could not kill process {proc.pid}: {e}")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

# 💬 Query handler
def query_llm(prompt, user_id="gradio-user"):
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "user_id": user_id,
        "prompt": prompt
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "❌ No response from LLM.")
        else:
            return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Exception: {str(e)}"

# 🎛️ Gradio Interface
iface = gr.Interface(
    fn=query_llm,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="LoRAAPI Chat UI",
    description="Talk to your fine-tuned local LLM running on FastAPI"
)

# 🚀 Launch
iface.launch(server_name="0.0.0.0", server_port=7863)
