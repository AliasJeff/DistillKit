"""Gradio interface for comparing teacher, student, and distilled student models."""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CONFIG
import time
import logging
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to store loaded models and tokenizers
MODELS: Dict[str, Optional[tuple]] = {"teacher": None, "student": None, "distilled_student": None}


def get_available_device() -> str:
    """Get the best available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_type: str, model_path: str) -> None:
    """Load a model and tokenizer if not already loaded.
    
    Args:
        model_type: One of 'teacher', 'student', or 'distilled_student'
        model_path: Path or name of the model to load
    """
    if model_type not in MODELS:
        raise ValueError(f"Invalid model type: {model_type}")

    if MODELS[model_type] is None:
        logger.info(f"Loading {model_type} model from {model_path}...")
        device = get_available_device()
        logger.info(f"Using device: {device}")

        # Determine torch dtype based on device
        if device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device == "mps":
            # MPS has better performance with float32
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float32

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device in ["cuda", "mps"] else None,
            trust_remote_code=True)

        # Move model to device if not using device_map
        if device in ["cuda", "mps"] and model.device.type != device:
            model = model.to(device)

        logger.info(f"Model loaded successfully on {device}")

        MODELS[model_type] = (model, tokenizer)
        logger.info(f"{model_type.capitalize()} model loaded successfully")


def generate_response(
    message: str,
    history: list,
    model_type: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate a response using the specified model.
    
    Args:
        message: User's input message
        history: Chat history (unused, but required by Gradio's ChatInterface)
        model_type: Type of model to use ('teacher', 'student', or 'distilled_student')
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated response text
    """
    if model_type not in MODELS or MODELS[model_type] is None:
        error_msg = f"Error: {model_type.capitalize()} model not loaded. Please load it first."
        return error_msg, history

    model, tokenizer = MODELS[model_type]
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

    # Format the prompt using the chat template
    messages = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    generation_time = time.time() - start_time

    # Add generation time to the response
    response = f"{response}\n\n[Generated in {generation_time:.2f}s]"
    return response, history


def chat_with_model(
    message: str,
    history: list,
    model_type: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Wrapper function for Gradio's ChatInterface."""
    response, _ = generate_response(
        message,
        history,
        model_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    # Add the user message and response to history in Gradio's expected format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def create_gradio_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    with gr.Blocks(title="Model Comparison") as demo:
        gr.Markdown("""
        # Model Comparison Interface
        Compare responses from the teacher, student, and distilled student models.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")
                model_selector = gr.Dropdown(choices=["Teacher", "Student", "Distilled Student"],
                                             value="Distilled Student",
                                             label="Select Model")

                gr.Markdown("### Generation Parameters")
                max_tokens = gr.Slider(minimum=10,
                                       maximum=2048,
                                       value=512,
                                       step=10,
                                       label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1,
                                        maximum=2.0,
                                        value=0.7,
                                        step=0.1,
                                        label="Temperature")
                top_p = gr.Slider(minimum=0.1,
                                  maximum=1.0,
                                  value=0.9,
                                  step=0.05,
                                  label="Top-p (nucleus sampling)")

                load_btn = gr.Button("Load Selected Model")
                load_status = gr.Textbox(value="No model loaded. Please load a model first.",
                                         label="Status",
                                         interactive=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=600, label="Chat with the selected model")
                msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
                clear = gr.ClearButton([msg, chatbot])

        def load_model_ui(model_name: str) -> str:
            """Load the selected model and update status."""
            try:
                model_type = model_name.lower().replace(" ", "_")
                if model_type == "distilled_student":
                    model_path = "./results"
                else:
                    model_path = CONFIG["models"][model_type]
                if not model_path:
                    return f"Error: No path configured for {model_name}"

                load_model(model_type, model_path)
                return f"{model_name} model loaded successfully!"
            except Exception as e:
                return f"Error loading {model_name}: {str(e)}"

        # Connect the load button
        load_btn.click(fn=load_model_ui, inputs=[model_selector], outputs=load_status)

        # Connect the chat interface
        msg.submit(fn=chat_with_model,
                   inputs=[
                       msg, chatbot,
                       gr.Textbox(value=lambda: model_selector.value.lower().replace(" ", "_"),
                                  visible=False), max_tokens, temperature, top_p
                   ],
                   outputs=[msg, chatbot])

        # Add some example prompts
        examples = gr.Examples(examples=[
            "Explain the concept of knowledge distillation in simple terms.",
            "What are the main differences between the teacher and student models?",
            "Can you summarize how this model was trained?"
        ],
                               inputs=msg,
                               label="Example Prompts")

    return demo


def main():
    """Launch the Gradio interface."""
    try:
        load_model("distilled_student", "./results")
    except Exception as e:
        logger.warning(f"Could not load distilled student model: {e}")

    # Create and launch the interface
    logger.info("Launching Gradio interface...")
    demo = create_gradio_interface()
    demo.launch(share=False, port=CONFIG["gradio"]["port"])


if __name__ == "__main__":
    main()
