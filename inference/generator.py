import torch
import tiktoken

from pathlib import Path
import sys
# Ensure project root is on sys.path so local package imports work when running this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DeepSeekV3

def generate_text(model_path, config, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = DeepSeekV3(config)

    # load checkpoint and try to auto-detect saved state dict / config
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # if checkpoint contains a config dict, prefer it
    ckpt_config = checkpoint.get("config", None)
    if ckpt_config is not None:
        print("Checkpoint contains a config — consider using it to construct the model:")
        print(ckpt_config)

    # Try loading and print mismatches (use strict=False for diagnostics)
    load_result = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint: missing keys:", load_result.missing_keys)
    print("Loaded checkpoint: unexpected keys:", load_result.unexpected_keys)

    model = model.to(device)
    model.eval()
    
    # Tokenize input
    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Decode and return
    result = enc.decode(generated.squeeze().tolist())
    return result

def run_inference_examples():
    """Run inference examples with different prompts."""
    try:
        from models import DeepSeekConfig
        
        config = DeepSeekConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=8,
            n_head=8,
            n_embd=512,
            kv_lora_rank=128,
            q_lora_rank=192,
            n_experts=8,
            n_experts_per_token=2,
            mtp_num_heads=1,
            dropout=0.1
        )
        
        test_prompts = [
            "Once upon a time",
            "The little girl",
            "In a magical forest",
            "The brave knight"
        ]
        
        print("=" * 50)
        print("DEEPSEEK-V3 INFERENCE EXAMPLES")
        print("=" * 50)
        
        for prompt in test_prompts:
            result = generate_text(
                "best_deepseek_v3.pt", 
                config, 
                prompt, 
                max_tokens=80, 
                temperature=0.7, 
                top_k=40
            )
            
            print(f"\nPrompt: '{prompt}'")
            print("Generated:", result)
            print("-" * 30)
            
    except FileNotFoundError:
        print("Model file 'best_deepseek_v3.pt' not found. Please train the model first.")
    except Exception as e:
        print(f"Error during inference: {e}")
    
if __name__ == "__main__":
    run_inference_examples()