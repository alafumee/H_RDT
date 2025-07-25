#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time

import torch
import yaml
from tqdm import tqdm

import sys
sys.path.append("/share/hongzhe/rdtv/")
from models.encoder.t5_encoder import T5Embedder


# Configuration parameters
MODEL_PATH = "/data/lingxuan/weights/t5-v1_1-xxl"
CONFIG_PATH = "/share/hongzhe/rdtv/configs/rdtv.yaml"
# ROOT_DIR = "/share/hongzhe/datasets/robotwin2/dataset/aloha-agilex"
ROOT_DIR = "/share/hongzhe/RoboTwin/data"
# No need for separate OUTPUT_DIR, save directly in the same directory as JSON files

# Multiprocessing parameters
NUM_GPUS = 8
PROCESSES_PER_GPU = 6
TOTAL_PROCESSES = NUM_GPUS * PROCESSES_PER_GPU

# Note: if your GPU VRAM is less than 24GB, 
# it is recommanded to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def find_all_instruction_files(root_dir):
    """
    Find all instruction JSON files
    
    Returns:
        list: List of instruction file paths
    """
    pattern = os.path.join(root_dir, "**/instructions/*.json")
    instruction_files = glob.glob(pattern, recursive=True)
    return sorted(instruction_files)

def extract_task_and_episode(instruction_path):
    """
    Extract task name and episode name from instruction file path
    
    Args:
        instruction_path (str): Instruction file path
    
    Returns:
        tuple: (task_name, episode_name)
    """
    path_parts = Path(instruction_path).parts
    # Find the location of 'aloha-agilex'
    aloha_idx = None
    for i, part in enumerate(path_parts):
        if part == 'aloha-agilex':
            aloha_idx = i
            break
    
    if aloha_idx is not None and aloha_idx + 1 < len(path_parts):
        task_name = path_parts[aloha_idx + 1]  # Task name like 'adjust_bottle'
        episode_name = Path(instruction_path).stem  # Episode filename like 'episode0'
        return task_name, episode_name
    else:
        # Fallback approach
        return Path(instruction_path).parent.parent.parent.name, Path(instruction_path).stem

def load_instructions_from_json(json_path):
    """
    Load instructions from JSON file
    
    Args:
        json_path (str): JSON file path
    
    Returns:
        list: List of instructions
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        instructions = []
        
        # Load 'seen' instructions
        if 'seen' in data:
            instructions.extend(data['seen'])
        
        # Load 'unseen' instructions
        if 'unseen' in data:
            instructions.extend(data['unseen'])
        
        return instructions
        
    except Exception as e:
        print(f"Failed to read JSON file {json_path}: {e}")
        return []

def encode_instructions_on_gpu(args_tuple):
    """
    Wrapper function for encoding instructions on specified GPU
    
    Args:
        args_tuple: (instruction_files, gpu_id, process_id) tuple
    
    Returns:
        tuple: (success_count, total_count, gpu_id, process_id)
    """
    instruction_files, gpu_id, process_id = args_tuple
    
    try:
        return encode_instructions_worker(instruction_files, gpu_id, process_id)
    except Exception as e:
        print(f"GPU {gpu_id} process {process_id} error: {e}")
        return 0, len(instruction_files), gpu_id, process_id

def encode_instructions_worker(instruction_files, gpu_id, process_id):
    """
    Worker function for encoding instructions
    
    Args:
        instruction_files (list): List of instruction files to process
        gpu_id (int): GPU ID
        process_id (int): Process ID
    
    Returns:
        tuple: (success_count, total_count, gpu_id, process_id)
    """
    if not instruction_files:
        return 0, 0, gpu_id, process_id
    
    try:
        # Load config
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        
        # Set device
        device = torch.device(f"cuda:{gpu_id}")
        
        # Initialize T5 encoder
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device,
            use_offload_folder=OFFLOAD_DIR
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        
        success_count = 0
        
        # Process each instruction file
        for instruction_file in instruction_files:
            try:
                # Extract task name and episode name
                task_name, episode_name = extract_task_and_episode(instruction_file)
                
                # Load instructions
                instructions = load_instructions_from_json(instruction_file)
                
                if not instructions:
                    print(f"GPU {gpu_id} process {process_id}: Skip {instruction_file} - no instructions found")
                    continue
                
                print(f"GPU {gpu_id} process {process_id}: Processing {task_name}/{episode_name} - {len(instructions)} instructions")
                
                # Encode all instructions
                embeddings_list = []
                instructions_list = []
                
                for instruction in instructions:
                    if not instruction.strip():  # Skip empty instructions
                        continue
                    
                    # Encode instruction
                    tokens = tokenizer(
                        instruction, return_tensors="pt",
                        padding="longest",
                        truncation=True
                    )["input_ids"].to(device)
                    
                    tokens = tokens.view(1, -1)
                    with torch.no_grad():
                        embedding = text_encoder(tokens).last_hidden_state.detach().cpu()
                    
                    embeddings_list.append(embedding.squeeze(0))  # Remove batch dimension [seq_len, hidden_dim]
                    instructions_list.append(instruction)
                
                if not embeddings_list:
                    print(f"GPU {gpu_id} process {process_id}: Skip {instruction_file} - no valid instructions")
                    continue
                
                # Prepare data to save - keep embeddings as list without concatenation
                save_data = {
                    "task_name": task_name,
                    "episode_name": episode_name,
                    "instructions": instructions_list,
                    "embeddings": embeddings_list,  # Keep as list, each element is [seq_len, hidden_dim]
                    "num_instructions": len(instructions_list)
                }
                
                # Save in the same directory as JSON file
                instruction_dir = os.path.dirname(instruction_file)
                save_path = os.path.join(instruction_dir, f"{episode_name}.pt")
                torch.save(save_data, save_path)
                
                success_count += 1
                print(f"GPU {gpu_id} process {process_id}: Successfully saved {len(instructions_list)} instructions to {save_path}")
                
            except Exception as e:
                print(f"GPU {gpu_id} process {process_id}: Error processing {instruction_file}: {e}")
                continue
        
        return success_count, len(instruction_files), gpu_id, process_id
        
    except Exception as e:
        print(f"GPU {gpu_id} process {process_id}: Initialization failed: {e}")
        return 0, len(instruction_files), gpu_id, process_id

def distribute_files_to_processes(instruction_files, total_processes):
    """
    Distribute files to different processes
    
    Args:
        instruction_files (list): List of all instruction files
        total_processes (int): Total number of processes
    
    Returns:
        list: List of files for each process to handle
    """
    files_per_process = len(instruction_files) // total_processes
    remainder = len(instruction_files) % total_processes
    
    process_files = []
    start_idx = 0
    
    for i in range(total_processes):
        # First 'remainder' processes handle one additional file
        end_idx = start_idx + files_per_process + (1 if i < remainder else 0)
        process_files.append(instruction_files[start_idx:end_idx])
        start_idx = end_idx
    
    return process_files

def main():
    print(f"Searching for instruction files: {ROOT_DIR}")
    
    # Find all instruction files
    instruction_files = find_all_instruction_files(ROOT_DIR)
    
    if not instruction_files:
        print("No instruction files found")
        return
    
    print(f"Found {len(instruction_files)} instruction files")
    
    # Distribute files to processes
    process_files_list = distribute_files_to_processes(instruction_files, TOTAL_PROCESSES)
    
    # Prepare arguments
    args_list = []
    for i in range(TOTAL_PROCESSES):
        gpu_id = i // PROCESSES_PER_GPU
        process_id = i % PROCESSES_PER_GPU
        args_list.append((process_files_list[i], gpu_id, process_id))
    
    print(f"Using {NUM_GPUS} GPUs, {PROCESSES_PER_GPU} processes per GPU, total {TOTAL_PROCESSES} processes")
    
    # Show workload for each process
    for i, (files, gpu_id, process_id) in enumerate(args_list):
        print(f"Process {i}: GPU {gpu_id} process {process_id} - {len(files)} files")
    
    # Record start time
    start_time = time.time()
    
    # Start multiprocessing
    with mp.Pool(processes=TOTAL_PROCESSES) as pool:
        results = pool.map(encode_instructions_on_gpu, args_list)
    
    # Collect results
    total_success = sum(result[0] for result in results)
    total_files = sum(result[1] for result in results)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print(f"\nCompleted! Successfully processed {total_success}/{total_files} instruction files")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average per file: {elapsed_time/total_files:.2f} seconds")
    
    # Show detailed results for each process
    print("\nProcessing results for each process:")
    for i, (success, total, gpu_id, process_id) in enumerate(results):
        print(f"Process {i} (GPU {gpu_id} process {process_id}): {success}/{total} files")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure proper GPU context passing
    main() 