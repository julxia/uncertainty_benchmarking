import time
import sys
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM

def format_timestamp(timestamp):
    """Format Unix timestamp to YYYY/MM/DD HH:MM:SS.mmm format"""
    return datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        print("ERROR: No GPU found, returning...")
        return
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Running inference on {device}")

    print("Loading model and processor...")
    model_name = "microsoft/git-large-coco"
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Loading dataset...")
    try:
        dataset = load_dataset("AIEnergyScore/image_captioning", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Running inference on {len(dataset)} examples...")
    start_time = time.time()

    total_tokens_generated = 0
    batch_size = 32

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        images = batch["image"]
        
        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(device)
        
        with torch.no_grad():  
            outputs = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        
        for output in outputs:
            tokens_generated = output.shape[0]
            total_tokens_generated += tokens_generated
        
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
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Tokens per second: {total_tokens_generated / elapsed_time:.2f}")

if __name__ == "__main__":
    run_inference()