import time
import sys
import torch
from datetime import datetime
from datasets import load_dataset
from diffusers import AutoPipelineForText2Image

def format_timestamp(timestamp):
    """Format Unix timestamp to YYYY/MM/DD HH:MM:SS.mmm"""
    return datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
    
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("ERROR: No GPU found, returning...")
        return

    if device.type == "cuda":
        torch.cuda.synchronize()


    print(f"Running inference on {device}")

    print("Loading model...")
    model_name = "stabilityai/sdxl-turbo"
    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Loading dataset...")

    try:
        dataset = load_dataset("AIEnergyScore/image_generation", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Running inference on {len(dataset)} examples...")
    start_time = time.time()

    total_prompts = 0
    batch_size = 8 

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        prompts = batch["prompt"]
        
        for prompt in prompts:
            with torch.no_grad():
                _ = pipeline(
                    prompt=prompt,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    output_type="pil"
                )
            
            total_prompts += 1
        
        if (i // batch_size) % 10 == 0:
            progress = min(i + batch_size, len(dataset)) / len(dataset) * 100
            print(f"Progress: {progress:.1f}% ({i + batch_size}/{len(dataset)})")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nInference completed!")
    print(f"START: {format_timestamp(start_time)}, END: {format_timestamp(end_time)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total prompts processed: {total_prompts}")
    print(f"Images per second: {total_prompts / elapsed_time:.2f}")

if __name__ == "__main__":
    run_inference()