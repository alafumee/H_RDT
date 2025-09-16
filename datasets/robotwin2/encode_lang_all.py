#!/usr/bin/env python3
import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob

# Add the project root to sys.path for importing models
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.encoder.t5_encoder import T5Embedder

# Assuming T5Embedder is available from your existing code
# from your_module import T5Embedder

class T5InstructionEncoder:
    def __init__(self, model_path, device="cuda", max_length=512):
        """Initialize T5 embedder for encoding instructions."""
        try:
            self.text_embedder = T5Embedder(
                from_pretrained=model_path,
                model_max_length=max_length,
                device=device,
                use_offload_folder=None
            )
            self.tokenizer = self.text_embedder.tokenizer
            self.text_encoder = self.text_embedder.model
            self.device = device
            print("T5 embedder initialized successfully")
        except Exception as e:
            print(f"Error initializing T5 embedder: {e}")
            raise

    def encode_instructions(self, instructions):
        """Encode a list of instructions and return embeddings."""
        if not instructions:
            return []
        
        embeddings = []
        
        for instruction in instructions:
            try:
                # Tokenize instruction
                tokens = self.tokenizer(
                    instruction,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True
                )["input_ids"].to(self.device)
                
                tokens = tokens.view(1, -1)
                
                # Generate embeddings
                with torch.no_grad():
                    embedding = self.text_encoder(tokens).last_hidden_state.detach().cpu()
                
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Warning: Error encoding instruction '{instruction[:50]}...': {e}")
                # Add a zero embedding as placeholder
                embeddings.append(torch.zeros(1, tokens.shape[1] if 'tokens' in locals() else 512, self.text_encoder.config.d_model))
        
        return embeddings

def find_all_episodes(dataset_root="/dataset"):
    """Find all episode JSON files in the dataset."""
    dataset_path = Path(dataset_root)
    episode_files = []
    
    # Find all instruction JSON files
    pattern = str(dataset_path / "*/aloha-agilex_*/instructions/episode*.json")
    json_files = glob.glob(pattern)
    
    for json_file in json_files:
        path_parts = Path(json_file).parts
        
        # Extract task name, data type, and episode ID
        task_name = None
        data_type = None
        episode_file = Path(json_file).name
        
        for i, part in enumerate(path_parts):
            if part.startswith("aloha-agilex_"):
                data_type = part
                if i > 0:
                    task_name = path_parts[i-1]
                break
        
        if task_name and data_type:
            episode_files.append({
                'task_name': task_name,
                'data_type': data_type,
                'episode_file': episode_file,
                'full_path': json_file
            })
    
    return episode_files

def process_dataset_instructions(dataset_root="/dataset", 
                               model_path=None,
                               output_dir=None,
                               device="cuda",
                               max_length=512):
    """Process all instruction files in the dataset."""
    
    # Set default paths
    if model_path is None:
        model_path = os.getenv("T5_MODEL_PATH", "/scratch/yz12129/t5_model")
    
    if output_dir is None:
        output_dir = Path(dataset_root).parent / "encoded_instructions"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Dataset root: {dataset_root}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Initialize encoder
    encoder = T5InstructionEncoder(model_path, device, max_length)
    
    # Find all episode files
    episode_files = find_all_episodes(dataset_root)
    print(f"Found {len(episode_files)} episode files")
    
    # Group by task and episode
    episodes_by_task = {}
    for file_info in episode_files:
        task_name = file_info['task_name']
        episode_file = file_info['episode_file']
        
        if task_name not in episodes_by_task:
            episodes_by_task[task_name] = {}
        
        if episode_file not in episodes_by_task[task_name]:
            episodes_by_task[task_name][episode_file] = []
        
        episodes_by_task[task_name][episode_file].append(file_info)
    
    total_episodes = sum(len(episodes) for episodes in episodes_by_task.values())
    print(f"Processing {total_episodes} unique episodes across {len(episodes_by_task)} tasks")
    
    # Process each episode
    processed_count = 0
    error_count = 0
    
    for task_name, episodes in episodes_by_task.items():
        print(f"\nProcessing task: {task_name}")
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(exist_ok=True)
        
        for episode_file, file_infos in tqdm(episodes.items(), desc=f"Episodes in {task_name}"):
            episode_id = episode_file.replace('.json', '')
            
            save_path = task_output_dir / f"{episode_id}_encoded.pt"
            if save_path.exists():
                print(f"  Skipping {episode_id}, already processed")
                continue

            try:
                # Collect all instructions for this episode
                all_instructions = []
                episode_metadata = {
                    'task_name': task_name,
                    'episode_id': episode_id,
                    'data_sources': []
                }
                
                for file_info in file_infos:
                    # Load JSON file
                    with open(file_info['full_path'], 'r') as f:
                        data = json.load(f)
                    
                    # Extract instructions
                    instructions = []
                    if isinstance(data, dict):
                        # Handle both "seen"/"unseen" format and direct list format
                        if 'seen' in data and 'unseen' in data:
                            instructions.extend(data['seen'])
                            instructions.extend(data['unseen'])
                        elif 'instructions' in data:
                            instructions.extend(data['instructions'])
                        elif isinstance(data, list):
                            instructions.extend(data)
                        else:
                            # Try to extract any list values
                            for key, value in data.items():
                                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                    instructions.extend(value)
                    
                    all_instructions.extend(instructions)
                    
                    # Add metadata
                    episode_metadata['data_sources'].append({
                        'data_type': file_info['data_type'],
                        'file_path': file_info['full_path'],
                        'instruction_count': len(instructions)
                    })
                
                # Remove duplicates while preserving order
                unique_instructions = []
                seen_instructions = set()
                for instruction in all_instructions:
                    if instruction not in seen_instructions:
                        unique_instructions.append(instruction)
                        seen_instructions.add(instruction)
                
                print(f"  {episode_id}: {len(all_instructions)} total, {len(unique_instructions)} unique instructions")
                
                if not unique_instructions:
                    print(f"  Warning: No instructions found for {episode_id}")
                    continue
                
                # Encode instructions
                embeddings = encoder.encode_instructions(unique_instructions)
                
                # Prepare data to save
                episode_data = {
                    'metadata': episode_metadata,
                    'instructions': unique_instructions,
                    'embeddings': embeddings,
                    'instruction_count': len(unique_instructions),
                    'embedding_shapes': [emb.shape for emb in embeddings] if embeddings else []
                }
                
                # Save encoded episode
                # save_path = task_output_dir / f"{episode_id}_encoded.pt"
                torch.save(episode_data, save_path)
                
                print(f"  ✓ Saved {len(embeddings)} encodings to {save_path}")
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {episode_id}: {e}")
                error_count += 1
                continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} episodes")
    print(f"Errors: {error_count} episodes")
    print(f"Output directory: {output_dir}")

def load_encoded_episode(file_path):
    """Helper function to load and inspect encoded episode data."""
    data = torch.load(file_path, map_location='cpu')
    
    print(f"Episode: {data['metadata']['episode_id']}")
    print(f"Task: {data['metadata']['task_name']}")
    print(f"Instructions: {data['instruction_count']}")
    print(f"Data sources: {len(data['metadata']['data_sources'])}")
    
    if data['embeddings']:
        print(f"Embedding shapes: {data['embedding_shapes'][:3]}...")  # Show first 3
    
    return data

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "/dataset"
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        process_dataset_instructions(
            dataset_root=dataset_root,
            model_path=model_path, 
            output_dir=output_dir,
            device=device
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise