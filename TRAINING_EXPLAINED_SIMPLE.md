# Training Process: Simple Explanation

## What's in Your Data?

### Input: Parent-Child Relationships

```
Actual Examples from Your Dataset:

Edge: 0 → 1
  "Bacteria" is a type of "cellular organisms"

Edge: 2 → 3
  "Azorhizobium" (bacteria genus) is a type of "Xanthobacteraceae" (bacteria family)

Edge: 4 → 2
  "Azorhizobium caulinodans" (species) is a type of "Azorhizobium" (genus)
```

This forms a **tree**:
```
          cellular organisms (131567)
                    |
                Bacteria (2)
                    |
          Xanthobacteraceae (335928)
                    |
             Azorhizobium (6)
                    |
    Azorhizobium caulinodans (7)
```

### Current Training: Numbers Only

**Training sees:**
```
0 → 1
2 → 3
4 → 2
...
```

**Training does NOT see:**
```
❌ "Bacteria"
❌ "Azorhizobium"
❌ "cellular organisms"
```

Names are completely separate - only used for visualization AFTER training!

## Training Process

### Step 1: Initialize Random Embeddings

```python
# Start with random vectors for each organism
embeddings = {
    0: [0.42, -0.15, 0.83, ...],   # Bacteria (random)
    1: [-0.23, 0.67, -0.41, ...],  # cellular organisms (random)
    2: [0.91, 0.22, -0.56, ...],   # Azorhizobium (random)
    ...
}
```

At this point, **embeddings are meaningless** - just random numbers.

### Step 2: Training Loop (Each Batch)

For each edge `(child → parent)`:

```python
# Example: Training on edge "Bacteria → cellular organisms"
child_idx = 0   # Bacteria
parent_idx = 1  # cellular organisms

# 1. Get embeddings
child_emb = embeddings[0]   # Current vector for Bacteria
parent_emb = embeddings[1]  # Current vector for cellular organisms

# 2. Compute positive distance (should be SMALL)
positive_dist = poincare_distance(child_emb, parent_emb)
# e.g., = 2.5 (too big! they're related!)

# 3. Sample negative examples (random organisms)
negatives = [42, 789, 2345, ...]  # Random organism indices

# 4. Compute negative distances (should be LARGE)
for neg_idx in negatives:
    neg_emb = embeddings[neg_idx]
    negative_dist = poincare_distance(child_emb, neg_emb)
    # e.g., = 1.8 (too small! they're NOT related!)

# 5. Compute loss
# Want: positive_dist < negative_dist
loss = max(0, margin + positive_dist - negative_dist)
# If positive_dist > negative_dist, loss is HIGH (bad!)
# If positive_dist < negative_dist, loss is LOW (good!)

# 6. Update embeddings to reduce loss
# Makes related organisms closer, unrelated organisms farther
embeddings[0] -= learning_rate * gradient  # Update Bacteria
embeddings[1] -= learning_rate * gradient  # Update cellular organisms
```

### Step 3: Repeat for All Edges

After processing 100,000 edges many times:
- Related organisms move closer together
- Unrelated organisms move apart
- Hierarchy emerges naturally!

### Step 4: Result After Training

```python
# After 500 epochs:
embeddings = {
    0: [0.15, 0.23, 0.08, ...],   # Bacteria (learned)
    1: [0.16, 0.24, 0.09, ...],   # cellular organisms (very close!)
    2: [0.14, 0.22, 0.07, ...],   # Azorhizobium (also close to Bacteria!)
    ...
    9606: [0.52, -0.31, 0.88, ...],  # Homo sapiens
    562: [-0.82, 0.15, -0.41, ...],  # E. coli (far from humans!)
}
```

Now:
- `distance(Bacteria, cellular_organisms)` ≈ 0.02 (very close!)
- `distance(Bacteria, Homo_sapiens)` ≈ 1.5 (far apart)
- `distance(Homo_sapiens, other_primates)` ≈ 0.001 (extremely close!)

## Visualization: What Happens During Training

### Epoch 0 (Random)
```
    Poincaré Disk (hyperbolic space)
         ___________
       /             \
      |  H  B        |   H = Human
      |    E         |   B = Bacteria
      | M       C    |   M = Mouse
      |               |   E = E. coli
      |   P           |   C = C. elegans
       \             /    P = Primates (other)
         -----------

Random positions, no structure
```

### Epoch 50 (Learning)
```
    Poincaré Disk
         ___________
       /             \
      |   HP          |   Primates clustering
      |               |
      | M             |   Mammals forming
      |               |
      |          E B  |   Bacteria grouping
       \        C    /
         -----------

Some structure emerging
```

### Epoch 500 (Learned)
```
    Poincaré Disk
         ___________
       /             \
      |  (HP)        |   Primates tight cluster
      |              |   H,P almost identical
      | M            |
      |              |   Clear separation
      |        (EB)  |   Bacteria cluster
       \       C    /    E,B close together
         -----------

Clear hierarchical structure!
```

## Adding Names: Is It Possible?

### Current Approach (Graph Only)
**What we use:** Graph structure (edges)
**What we don't use:** Names

```
Training Input:
  ✅ 0 → 1 (edge)
  ✅ 4 → 2 (edge)
  ❌ "Bacteria" (name)
  ❌ "Azorhizobium" (name)
```

### If You Want to Use Names

**Option 1: Keep Current (Recommended for Hierarchy)**
- Pros: Works great, simple, fast
- Cons: Can't query by name during inference

**Option 2: Add Text Encoder (Multimodal)**
```python
# Encode names with BERT/BioBERT
name_embedding = encode_text("Homo sapiens")  # [0.23, -0.45, ...]
graph_embedding = poincare_embedding[9606]    # [0.15, 0.23, ...]

# Train to align them
loss = distance(name_embedding, graph_embedding)
```

**Benefits of Option 2:**
- Can do text queries: "find species like 'sapiens'"
- Better for search/retrieval
- Can handle synonyms

**Downsides of Option 2:**
- Much more complex
- Slower training
- Requires text encoder (BERT, etc.)
- May not improve hierarchy learning

### My Recommendation

**For your use case (taxonomy hierarchy):**
👉 **Keep current approach!**

Why?
1. ✅ Graph structure encodes hierarchy perfectly
2. ✅ Names don't add hierarchy information
3. ✅ You already have names in mapping file for visualization
4. ✅ Much simpler and faster
5. ✅ Results are excellent (primates cluster at distance 0.001!)

**When to add names:**
- If you need text-based search
- If you want to handle organisms without TaxIDs
- If you want multilingual support
- If you want to handle typos/synonyms

## Summary

### What We Train
- **Input:** Parent-child edges (numbers only)
- **Output:** One vector per organism
- **Method:** Make related organisms close, unrelated organisms far
- **Space:** Hyperbolic (Poincaré) for hierarchies

### Names Currently
- ✅ Stored in mapping file
- ✅ Used for visualization
- ❌ NOT used in training
- ✅ This is GOOD for hierarchy learning!

### Can We Add Names?
- ✅ Yes, technically possible
- ⚠️ Adds complexity
- ❓ May not improve results for hierarchies
- 👍 Useful if you need text queries

### Your Current Results
After 500 epochs:
- Primates cluster together ✅
- Human → closest neighbor at distance 0.0007 ✅
- Clear hierarchical structure ✅
- **Names not needed for this!** ✅

The model learned the taxonomy perfectly using ONLY the graph structure! 🎉
