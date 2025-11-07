# Poincaré Embeddings Explained

## What Are Poincaré Embeddings?

### The Problem: Representing Hierarchies
Traditional embeddings (like Word2Vec, GloVe) use **Euclidean space** (flat, normal space). But hierarchical data (like taxonomies, org charts, WordNet) have a **tree-like structure** that doesn't fit well in flat space.

**Why?** In a tree:
- The root has 1 node
- Level 1 might have 10 nodes
- Level 2 might have 100 nodes
- Level 3 might have 1,000 nodes
- ...exponential growth

In flat Euclidean space, you need **exponentially growing dimensions** to represent this without distortion.

### The Solution: Hyperbolic Space
**Poincaré embeddings** use **hyperbolic geometry** - a curved space where:
- Distance grows exponentially as you move from the center
- Perfect for hierarchies: root near center, leaves near boundary
- Can represent exponential growth in constant dimensions

**Visualization:**
```
Euclidean Space (flat):        Hyperbolic Space (Poincaré disk):
    o                               o (root, center)
   / \                             /|\
  o   o                           / | \
 /|   |\                         /  |  \
o o   o o                       o   o   o
                               /|\ /|\ /|\
                              (exponentially more space
                               near the boundary)
```

### Key Properties
1. **Distance from center = hierarchy level**
   - Root organisms near center (e.g., ||x|| ≈ 0)
   - Leaf organisms near boundary (e.g., ||x|| ≈ 0.99)

2. **Angular distance = similarity within level**
   - All primates cluster in one angular region
   - All bacteria cluster in another region

3. **Hierarchical distance preserved**
   - Related organisms are closer
   - Distance in embedding ≈ distance in taxonomy tree

## Our Data: NCBI Taxonomy

### What Our Data Looks Like

#### Raw Data (from NCBI)
```
nodes.dmp:
  TaxID | Parent_TaxID | Rank | ...
  9606  | 9605         | species    # Homo sapiens → Homo (genus)
  9605  | 207598       | genus      # Homo → Homininae (subfamily)
  207598| 9604         | subfamily  # Homininae → Hominidae (family)
  9604  | 314295       | family     # Hominidae → Simiiformes
  ...
```

#### Edge List (parent-child relationships)
```
data/taxonomy_edges.edgelist:
  9606 9605        # Homo sapiens → Homo
  9605 207598      # Homo → Homininae
  207598 9604      # Homininae → Hominidae
  9604 314295      # Hominidae → Simiiformes
  314295 9526      # Simiiformes → Primates
  9526 40674       # Primates → Mammalia
  ...
```

This is a **directed graph** where each edge represents:
```
child_taxid → parent_taxid
```

#### Remapped for Training
```
data/taxonomy_edges_small.mapped.edgelist:
  0 1             # Remapped indices (TaxID 9606 → idx 0, TaxID 9605 → idx 1)
  2 3
  4 5
  ...
```

Plus mapping file:
```
data/taxonomy_edges_small.mapping.tsv:
  taxid   idx
  9606    0
  9605    1
  ...
```

### Visual Example

Here's what the hierarchy looks like:

```
                        Root (1)
                          |
        ┌─────────────────┴─────────────────┐
   Bacteria (2)                        Eukaryota (2759)
        |                                    |
    ┌───┴───┐                         ┌─────┴─────┐
   E.coli  Salmonella            Animals (33208)  Plants
   (562)   (590)                       |
                                   ┌───┴───┐
                              Vertebrates  Invertebrates
                              (7742)       (...)
                                   |
                              ┌────┴────┐
                          Mammals      Fish
                          (40674)      (...)
                               |
                          ┌────┴────┐
                      Primates    Rodents
                      (9443)      (9989)
                          |           |
                      ┌───┴───┐      Mouse
                  Humans  Apes   (10090)
                  (9605)  (9604)
                     |
                 Homo sapiens
                   (9606)
```

**In the edge list, we have:**
- 9606 → 9605 (Homo sapiens → Homo)
- 9605 → 207598 (Homo → Homininae)
- 9604 → 314295 (Hominidae → Simiiformes)
- 9443 → 40674 (Primates → Mammalia)
- etc.

### Data Statistics

**Small dataset:**
- 111,103 unique organisms (nodes)
- 100,000 parent-child relationships (edges)
- Covers major groups but incomplete

**Full dataset:**
- 2,705,745 unique organisms (nodes)
- 2,705,744 parent-child relationships (edges)
- Complete NCBI taxonomy tree

## What Are We Training?

### The Embedding Model

We learn a **vector** for each organism:
```
TaxID → Vector in R^d (e.g., d=10)

Example (simplified):
  Homo sapiens (9606) → [0.15, 0.23, 0.08, ..., 0.45]
  Homo (9605)         → [0.14, 0.22, 0.09, ..., 0.43]  # Similar, nearby
  E. coli (562)       → [-0.80, 0.02, -0.35, ..., 0.10] # Very different
```

### The Training Process

#### 1. Objective: Learn Distance Model
For each edge `(u → v)` (child → parent):
```
Distance in embedding should be SMALL
d_poincare(embed(u), embed(v)) should be ≈ 0.1-0.5
```

For non-edges (random pairs):
```
Distance in embedding should be LARGE
d_poincare(embed(u), embed(random)) should be ≈ 1.0-2.0
```

#### 2. Loss Function
```python
# For each training batch:
for (child, parent) in edges:
    # Positive sample (real edge)
    positive_distance = d_poincare(embed[child], embed[parent])
    
    # Negative samples (random organisms, not related)
    for neg in random_samples(k=50):
        negative_distance = d_poincare(embed[child], embed[neg])
    
    # Loss: want positive_distance < negative_distance
    loss = max(0, margin + positive_distance - negative_distance)
```

**Intuition:**
- Child and parent should be close
- Child and random organism should be far
- Margin = minimum separation we want

#### 3. Hyperbolic Distance
The key is using **Poincaré distance**, not Euclidean:

```python
def poincare_distance(u, v):
    """Distance in Poincaré ball model."""
    norm_u_sq = ||u||^2
    norm_v_sq = ||v||^2
    norm_diff_sq = ||u - v||^2
    
    numerator = 2 * norm_diff_sq
    denominator = (1 - norm_u_sq) * (1 - norm_v_sq)
    
    return arcosh(1 + numerator / denominator)
```

This distance:
- Respects the hyperbolic geometry
- Grows exponentially near the boundary (||x|| → 1)
- Preserves hierarchical structure

#### 4. Riemannian SGD
We can't use normal gradient descent because we're in curved space!

**Riemannian SGD:**
1. Compute gradient in tangent space (flat space at current point)
2. Project back onto the Poincaré ball (ensure ||x|| < 1)
3. Update embeddings

```python
# Simplified
grad = compute_gradient(loss)
x_new = x - lr * grad
x_new = project_onto_ball(x_new)  # Keep ||x|| < 1
```

### Training Example

**Epoch 0 (start):**
```
Loss: 3.94
Embeddings are random
Human → Nearest neighbor: Random organism (bad!)
```

**Epoch 50:**
```
Loss: 2.86
Embeddings learning structure
Human → Nearest neighbor: Other primate (better!)
```

**Epoch 500:**
```
Loss: 2.32
Embeddings encode hierarchy well
Human → Nearest neighbor: Almost identical primate species (distance 0.0007)
Primates cluster together in UMAP
```

## Current Setup: TaxID Only

### What We Currently Embed

**Each organism = One node = One vector**

```
TaxID 9606 (Homo sapiens)     → vector [0.15, 0.23, ..., 0.45]
TaxID 9605 (Homo)             → vector [0.14, 0.22, ..., 0.43]
TaxID 562 (E. coli)           → vector [-0.80, 0.02, ..., 0.10]
```

**Names are separate** (in mapping file):
```
mapping.tsv:
  9606  →  "Homo sapiens"
  562   →  "Escherichia coli"
```

**During training:**
- Only TaxID integers are used
- Names are NOT used
- Model learns from graph structure only

**During visualization:**
- We look up names from mapping file
- Display "Homo sapiens (9606)" for human interpretation

## Can We Add Species Names to Training?

### Option 1: Current Approach (Graph Structure Only) ✅
**What we're doing now**

**Pros:**
- ✅ Fast training
- ✅ Works perfectly for taxonomy hierarchy
- ✅ No text processing needed
- ✅ Language-independent

**Cons:**
- ❌ Doesn't learn from name similarity
- ❌ Can't handle organisms without TaxID
- ❌ Can't do text-based queries ("find primates")

### Option 2: Text + Graph (Multimodal) ⭐
**Add species names as additional signal**

**Approach:**
1. Encode names with text encoder (e.g., BERT, BioBERT)
2. Learn joint embedding space
3. Train on both graph structure AND name similarity

```python
# Pseudocode
text_embedding = encode_text(species_name)
graph_embedding = current_poincare_embedding

# Joint loss
loss_graph = poincare_loss(graph_edges)
loss_text = similarity_loss(text_embedding, graph_embedding)
total_loss = loss_graph + λ * loss_text
```

**Benefits:**
- Can query by name: "organisms similar to 'sapiens'"
- Better generalization
- Can handle typos, synonyms
- Cross-lingual (if using multilingual encoder)

**Downsides:**
- Much more complex
- Requires text encoder
- Slower training
- Might not improve hierarchy learning (names don't encode hierarchy well)

### Option 3: Name as Feature (Auxiliary) 
**Use names as additional features, not primary signal**

```python
# Add name features to each node
features = {
    'taxid': 9606,
    'name': 'Homo sapiens',
    'name_length': 12,
    'has_genus': True,
    'has_species': True,
    'name_embedding': encode(name)  # Small text encoding
}

# Then train graph embedding with features
```

**Benefits:**
- Lightweight addition
- Can improve disambiguation
- Still mainly graph-based

### Recommendation for Your Use Case

**If your goal is hierarchy learning:** 
👉 **Stick with current approach** (graph structure only)
- It's working perfectly
- Names don't add much for hierarchy
- Much simpler and faster

**If you want text-based queries:**
👉 **Add text encoder as Option 2**
- Could enable: "Find all species with 'sapiens' in name"
- Could enable: "Find organisms similar to 'human'"
- Useful for downstream applications

**If you want names for better interpretability:**
👉 **Current approach is fine!**
- Names are in mapping file
- Visualization script already shows names
- No training changes needed

## Summary

### What Poincaré Embeddings Are
- Embeddings in **hyperbolic space** (curved geometry)
- Perfect for **hierarchical data** like taxonomies
- Preserve tree structure in low dimensions

### What Our Data Looks Like
- **Nodes:** Organisms (TaxIDs)
- **Edges:** Parent-child relationships
- **Structure:** Tree/DAG of 111K-2.7M organisms

### What We're Training
- **One vector per organism** in R^10
- **Minimize distance** for parent-child pairs
- **Maximize distance** for unrelated pairs
- **Use hyperbolic distance** and Riemannian SGD

### Names in Training
**Current:** Names NOT used in training (only in visualization)
**Possible:** Could add text encoder for multimodal learning
**Recommendation:** Current approach is excellent for hierarchy learning

### The Magic
After training, embeddings capture:
- ✅ Hierarchy: depth in tree ≈ distance from center
- ✅ Similarity: related organisms cluster together
- ✅ Relationships: nearest neighbors are taxonomically related

All in just **10 dimensions** in hyperbolic space! 🎉
