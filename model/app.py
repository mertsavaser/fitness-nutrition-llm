"""
Inference script for Fitness & Nutrition Coach LLM.
Deployed on Hugging Face Spaces with Gradio interface.

This script loads the base Qwen2.5-3B-Instruct model with LoRA adapters
and provides a Gradio interface for interactive inference.
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================
# CONFIGURATION
# =====================
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_ID = "mertsavaser/fitness-nutrition-qwen2.5-3b-lora"

SYSTEM_PROMPT = """You are a professional Fitness & Nutrition Coach.

Rules:
- Always answer in English.
- Use a friendly but confident professional coach tone.
- Give clear, structured, actionable advice.
- Do NOT mention that you are an AI.
- If the topic is medical, add a short general disclaimer.
"""

# =====================
# MODEL LOADING
# =====================
# Load tokenizer for Qwen2.5 model
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# Load base model with 4-bit quantization for efficient inference
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapters from Hugging Face Hub
model = PeftModel.from_pretrained(base_model, LORA_ID)
model.eval()

# =====================
# INFERENCE FUNCTION
# =====================
def chat(user_input: str) -> str:
    """
    Generate response to user input using the fine-tuned model.
    
    Args:
        user_input: User's fitness or nutrition question
        
    Returns:
        Model's response as a string
    """
    # Format messages with system prompt and user input
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_input.strip()}
    ]

    # Apply Qwen2.5 chat template (handles system/user/assistant roles)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,          # Maximum response length
            temperature=0.6,             # Sampling temperature for natural responses
            top_p=0.9,                  # Nucleus sampling
            do_sample=True,             # Enable sampling (prevents repetitive outputs)
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract only the newly generated tokens (exclude input prompt)
    generated_tokens = outputs[0][inputs.shape[-1]:]

    # Decode tokens to text
    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

# =====================
# GRADIO UI
# =====================
demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        label="Ask your fitness or nutrition question",
        lines=3,
        placeholder="e.g. Give me 3 tips to gain muscle"
    ),
    outputs=gr.Textbox(
        label="Coach Response",
        lines=10
    ),
    title="Fitness & Nutrition Coach (LoRA)",
    description="Fine-tuned Qwen2.5-3B LoRA model for practical fitness and nutrition guidance."
)

if __name__ == "__main__":
    demo.launch()
