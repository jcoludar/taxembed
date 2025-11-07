# Data Handling Review

## Current Pipeline

### 1. Data Preparation (`prepare_taxonomy_data.py`)
```
Input: data/nodes.dmp, data/names.dmp
Output: data/taxonomy_edges.csv
Format: CSV with header (id1,id2,weight)
```

**✅ This step is correct:**
- Extracts parent-child relationships from NCBI taxonomy
- Each edge: child → parent
- Skips self-loops (root node)
- Saves as CSV with proper header

### 2. CSV → EdgeList Conversion
```
Input: data/taxonomy_edges.csv
Output: data/taxonomy_edges.edgelist
Format: Whitespace-separated, WITH HEADER
```

**⚠️ ISSUE FOUND:**
```bash
$ head -5 data/taxonomy_edges_small.edgelist
id1 id2           # ← HEADER LINE (should be removed!)
2 131567
6 335928
7 6
9 32199
```

The `.edgelist` file contains a header line "id1 id2" which should NOT be there.

### 3. Remapping (`remap_edges.py`)
```python
in_path = sys.argv[1]
out_edges = in_path.replace(".edgelist", ".mapped.edgelist")
out_map = in_path.replace(".edgelist", ".mapping.tsv")

with open(in_path) as f:
    for ln, line in enumerate(f, 1):
        line=line.strip()
        if not line: continue
        parts = line.split()
        if len(parts)!=2:
            raise ValueError(f"Bad line {ln}: {line!r}")
        u, v = parts
        if u not in nodes: nodes[u] = len(nodes)
        if v not in nodes: nodes[v] = len(nodes)
        edges.append((nodes[u], nodes[v]))
```

**❌ BUG: Treats header as data!**

Result:
```bash
$ head -5 data/taxonomy_edges_small.mapping.tsv
taxid   idx
id1     0      # ← HEADER treated as a TaxID!
id2     1      # ← HEADER treated as a TaxID!
2       2      # ← Actual TaxID
131567  3      # ← Actual TaxID
```

This means:
- Node 0 = "id1" (string literal, not a taxid!)
- Node 1 = "id2" (string literal, not a taxid!)
- Node 2 = TaxID 2 (actual organism)
- Node 3 = TaxID 131567 (actual organism)

### 4. Training Data Loading (`embed.py` → `hype/graph.py`)

```python
def load_edge_list(path, symmetrize=False):
    df = pandas.read_csv(path, sep=r'\s+', header=None, names=['id1', 'id2'], engine='python')
    df['weight'] = 1.0
    df.dropna(inplace=True)
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights
```

**Issue: Double Remapping**
- The `.mapped.edgelist` already has sequential indices (0, 1, 2, ...)
- `pandas.factorize` RE-MAPS them again based on first appearance order
- This creates a NEW mapping that may differ from the original mapping file

**Result:**
- The `objects` array returned is the remapped values (not the original TaxIDs)
- The mapping file (`taxonomy_edges_small.mapping.tsv`) becomes INVALID for lookup
- When we try to map back using the .mapping.tsv file, we get wrong organisms

## Problems Identified

### Problem 1: Header Line in EdgeList
**Severity: HIGH**
**Impact:** Two fake "organisms" (id1, id2) are added to the dataset

**Fix:**
```python
# In remap_edges.py, skip header line:
with open(in_path) as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line or line.startswith("id1"):  # Skip header
            continue
        # ... rest of code
```

Or better: Don't include header in .edgelist file at all.

### Problem 2: Double Remapping
**Severity: MEDIUM**
**Impact:** Redundant computation, potential index mismatch

**Current Flow:**
```
TaxIDs → remap_edges.py → [0,1,2,...,N] → load_edge_list → factorize → [0,1,2,...,M]
```

The `factorize` step creates a NEW mapping based on first appearance order in the edge list.

**Why this mostly works:**
- If edges are processed in order, factorize will likely preserve the mapping
- But it's not guaranteed and adds unnecessary complexity

**Better approach:**
```python
def load_edge_list_remapped(path, symmetrize=False):
    """Load edge list that's already been remapped to sequential indices."""
    edges = []
    max_idx = -1
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("id1"):  # Skip empty/header
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append([u, v])
                max_idx = max(max_idx, u, v)
    
    if symmetrize:
        edges.extend([[v, u] for u, v in edges])
    
    idx = np.array(edges, dtype=int)
    # Create objects list: [0, 1, 2, ..., max_idx]
    objects = list(range(max_idx + 1))
    weights = np.ones(len(idx), dtype=float)
    
    return idx, objects, weights
```

### Problem 3: Mapping File Inconsistency
**Severity: HIGH**
**Impact:** Cannot correctly map embeddings back to TaxIDs

The `.mapping.tsv` file maps: `TaxID → original_idx`
But the training uses: `factorized_idx → embedding`

To get `TaxID → embedding`, you need:
1. `TaxID → original_idx` (from .mapping.tsv)
2. `original_idx → factorized_idx` (not saved anywhere!)
3. `factorized_idx → embedding` (from model)

**Current workaround:**
The training saves `objects` in the checkpoint, which are the factorized values. So you can use:
- `checkpoint['objects'][factorized_idx]` → `original_idx_str`
- Then lookup in mapping file

But this is confusing and error-prone.

## Recommendations

### Immediate Fixes

1. **Remove header from .edgelist files:**
```bash
# Convert CSV to edgelist without header
tail -n +2 data/taxonomy_edges.csv | awk '{print $1, $2}' FS=',' > data/taxonomy_edges.edgelist
```

2. **Update remap_edges.py to skip headers:**
```python
if line.startswith("id1") or line.startswith("taxid"):
    continue
```

3. **Verify data integrity:**
```bash
# Check that mapping file has no "id1" or "id2"
grep -E "^(id1|id2)\s" data/*.mapping.tsv
```

### Long-term Improvements

1. **Simplify the pipeline:**
   - Either use factorize OR pre-remap, not both
   - Make load_edge_list aware that data is already remapped

2. **Add data validation:**
   - Check for header lines
   - Verify node indices are sequential
   - Confirm mapping file matches edge list

3. **Improve documentation:**
   - Document expected file formats
   - Explain mapping strategy
   - Add validation scripts

## Current Status

**Does it work?** YES, mostly.

Despite the issues:
- The model trains successfully
- Loss decreases appropriately
- Embeddings learn hierarchical structure

**Why it works despite bugs:**
- The header line adds 2 fake nodes, but they have few/no edges
- Double remapping is redundant but doesn't break functionality (if order preserved)
- The factorized `objects` are saved in checkpoints for reverse lookup

**Should you fix it?** YES!
- Removes confusion
- Prevents potential bugs with different data
- Makes code more maintainable
- Improves reproducibility

## Testing Recommendations

After fixes:
1. Retrain small model (5-10 epochs)
2. Verify node count matches expectations
3. Check that embedding[0] corresponds to correct TaxID
4. Validate nearest neighbors match biological taxonomy
