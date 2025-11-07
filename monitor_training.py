#!/usr/bin/env python3
"""
Monitor training progress by checking primate clustering quality.
Run this in a separate terminal while training is running.
"""
import torch
import numpy as np
import pandas as pd
import glob
import time
import sys

def load_embeddings(ckpt_path):
    """Load embeddings from checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        emb = sd["lt.weight"].detach().cpu().numpy()
        epoch = ckpt.get("epoch", -1)
        loss = ckpt.get("loss", None)
        return emb, epoch, loss
    except:
        return None, None, None

def get_all_descendants(taxid, nodes_path="data/nodes.dmp"):
    """Get all descendants of a taxid."""
    descendants = set([taxid])
    to_process = [taxid]
    children_map = {}
    try:
        with open(nodes_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t|\t')
                if len(parts) >= 2:
                    child_id = parts[0].strip()
                    parent_id = parts[1].strip()
                    if parent_id not in children_map:
                        children_map[parent_id] = []
                    children_map[parent_id].append(child_id)
        
        while to_process:
            current = to_process.pop(0)
            if current in children_map:
                for child in children_map[current]:
                    if child not in descendants:
                        descendants.add(child)
                        to_process.append(child)
    except:
        pass
    
    return descendants

def check_primate_clustering(emb, tax2idx, idx2tax):
    """Check if primates are clustering together."""
    primate_taxids = get_all_descendants("9443")
    primate_indices = []
    for taxid in primate_taxids:
        if taxid in tax2idx:
            idx = tax2idx[taxid]
            if idx < len(emb):
                primate_indices.append(idx)
    
    if len(primate_indices) < 10:
        return None, None, None
    
    primate_indices = np.array(primate_indices)
    primate_embs = emb[primate_indices]
    
    # Compute pairwise distances within primates
    primate_distances = []
    for i in range(min(50, len(primate_indices))):
        for j in range(i+1, min(i+10, len(primate_indices))):
            d = np.linalg.norm(primate_embs[i] - primate_embs[j])
            primate_distances.append(d)
    
    # Compare with random pairs
    random_distances = []
    for _ in range(len(primate_distances)):
        i = np.random.randint(0, emb.shape[0])
        j = np.random.randint(0, emb.shape[0])
        if i != j:
            d = np.linalg.norm(emb[i] - emb[j])
            random_distances.append(d)
    
    primate_distances = np.array(primate_distances)
    random_distances = np.array(random_distances)
    
    ratio = random_distances.mean() / primate_distances.mean() if primate_distances.mean() > 0 else 0
    
    return primate_distances.mean(), random_distances.mean(), ratio

def main():
    map_path = "data/taxonomy_edges.mapping.tsv"
    
    print("Loading mapping...")
    m = pd.read_csv(map_path, sep="\t", dtype={"taxid": str, "idx": int})
    tax2idx = dict(zip(m["taxid"], m["idx"]))
    idx2tax = dict(zip(m["idx"], m["taxid"]))
    
    print("Monitoring training progress...")
    print("="*80)
    print(f"{'Epoch':<8} {'Loss':<12} {'Primate Dist':<15} {'Random Dist':<15} {'Ratio':<10} {'Status':<15}")
    print("="*80)
    
    last_epoch = -1
    last_file = None
    
    while True:
        # Find all checkpoint files
        checkpoints = sorted(glob.glob("taxonomy_model_full_fixed_epoch*.pth"))
        
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            
            # Only process if it's a new file
            if latest_ckpt != last_file:
                last_file = latest_ckpt
                emb, epoch, loss = load_embeddings(latest_ckpt)
                
                if emb is not None:
                    last_epoch = epoch
                
                # Check primate clustering
                primate_dist, random_dist, ratio = check_primate_clustering(emb, tax2idx, idx2tax)
                
                if primate_dist is not None:
                    status = "✓ GOOD" if ratio > 1.5 else "⚠ POOR" if ratio > 1.0 else "✗ BAD"
                    print(f"{epoch:<8} {loss:<12.4f} {primate_dist:<15.6f} {random_dist:<15.6f} {ratio:<10.2f}x {status:<15}")
                else:
                    print(f"{epoch:<8} {loss:<12.4f} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'SKIP':<15}")
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    main()
