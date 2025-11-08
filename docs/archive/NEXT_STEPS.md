# Next Steps: Extensions & Enhancements

## ✅ Current Status

**Repository is production-ready:**
- ✅ Clean, restructured codebase
- ✅ Data bugs fixed (111,103 clean nodes)
- ✅ Training validated (500 epochs, excellent results)
- ✅ Universal visualization tool
- ✅ Comprehensive documentation
- ✅ Updated README with usage-first approach

**Current Capabilities:**
- Train on NCBI taxonomy graph structure
- Learn hierarchical embeddings in hyperbolic space
- Visualize any taxonomic group
- Nearest neighbor queries
- Data validation

## 🚀 Proposed Extensions

### 1. Species Names Integration

**Goal:** Enable text-based queries and better organism search

**Current State:**
- Names stored in mapping file
- NOT used in training
- Only for visualization

**Extension:**
```python
# Multi-modal training
class MultiModalPoincare:
    def __init__(self):
        self.graph_encoder = PoincareEmbedding()  # Current
        self.text_encoder = BertModel()  # NEW
        
    def forward(self, taxid, name_text):
        graph_emb = self.graph_encoder(taxid)
        text_emb = self.text_encoder(name_text)
        
        # Align embeddings
        loss = distance(graph_emb, text_emb)
        return loss
```

**Benefits:**
- Query: "find species similar to 'sapiens'"
- Handle synonyms: "Escherichia coli" = "E. coli"
- Cross-lingual: search in any language
- Better disambiguation

**Implementation Steps:**
1. Load species names from `names.dmp`
2. Add BERT/BioBERT text encoder
3. Create text dataset: `(taxid, name)` pairs
4. Joint training: graph loss + text alignment loss
5. Update visualization to support text queries

**Effort:** Medium (1-2 weeks)

---

### 2. Protein Embeddings

**Goal:** Incorporate functional/sequence information

**Current State:**
- Only taxonomy structure
- No molecular information

**Extension:**
```python
# Add protein-level features
class ProteinEnhancedEmbedding:
    def __init__(self):
        self.taxonomy_encoder = PoincareEmbedding()  # Current
        self.protein_encoder = ESMModel()  # NEW (protein language model)
        self.fusion = FusionLayer()  # Combine both
        
    def forward(self, taxid, protein_sequences):
        tax_emb = self.taxonomy_encoder(taxid)
        
        # Aggregate proteins for organism
        protein_embs = [self.protein_encoder(seq) for seq in protein_sequences]
        org_protein_emb = aggregate(protein_embs)  # Mean/max pooling
        
        # Combine
        combined_emb = self.fusion(tax_emb, org_protein_emb)
        return combined_emb
```

**Data Sources:**
- UniProt for protein sequences
- KEGG for protein functions
- RefSeq for reference proteomes

**Benefits:**
- Find organisms by protein function
- Cluster by functional similarity
- Better for organisms with horizontal gene transfer
- Useful for drug discovery

**Implementation Steps:**
1. Download protein sequences from UniProt
2. Compute ESM/ProtT5 embeddings for each protein
3. Aggregate per organism (mean pooling)
4. Create fusion architecture
5. Joint training on taxonomy + proteins

**Effort:** High (3-4 weeks)

---

### 3. Additional Features

**Goal:** Incorporate phenotypic/genomic metadata

**Current State:**
- Only taxonomy relationships
- No organism features

**Extension:**
```python
# Add feature vectors
class FeatureAugmentedEmbedding:
    def __init__(self):
        self.taxonomy_encoder = PoincareEmbedding()  # Current
        self.feature_encoder = FeatureNet()  # NEW
        
    def forward(self, taxid, features):
        tax_emb = self.taxonomy_encoder(taxid)
        
        # Features: [genome_size, gc_content, temperature, ...]
        feat_emb = self.feature_encoder(features)
        
        # Concatenate or fuse
        combined = torch.cat([tax_emb, feat_emb], dim=-1)
        return combined
```

**Possible Features:**
- **Genomic:** genome size, GC content, chromosome count
- **Environmental:** temperature range, pH range, habitat
- **Morphological:** cell shape, motility, gram staining
- **Metabolic:** aerobic/anaerobic, carbon source

**Data Sources:**
- NCBI BioSample
- IMG (DOE Joint Genome Institute)
- BacDive (bacterial metadata)

**Benefits:**
- Predict missing features
- Find organisms by phenotype
- Better for ecological studies
- Feature-based queries

**Implementation Steps:**
1. Collect feature data from databases
2. Create feature vectors per organism
3. Add feature encoder network
4. Train with feature prediction as auxiliary task
5. Enable feature-based search

**Effort:** Medium (2-3 weeks)

---

### 4. Word Descriptions

**Goal:** Natural language descriptions and semantic search

**Current State:**
- No text descriptions
- No literature linkage

**Extension:**
```python
# Add descriptions
class DescriptionEnhancedEmbedding:
    def __init__(self):
        self.taxonomy_encoder = PoincareEmbedding()  # Current
        self.description_encoder = SentenceBERT()  # NEW
        
    def forward(self, taxid, description_text):
        tax_emb = self.taxonomy_encoder(taxid)
        
        # Encode description: "Gram-negative bacterium found in..."
        desc_emb = self.description_encoder(description_text)
        
        # Align
        loss = distance(tax_emb, desc_emb)
        return loss
```

**Data Sources:**
- Wikipedia organism articles
- NCBI organism descriptions
- Literature abstracts (PubMed)
- Textbooks and databases

**Benefits:**
- Semantic search: "find bacteria that ferment lactose"
- Generate organism descriptions
- Link to scientific literature
- Educational applications

**Implementation Steps:**
1. Scrape/download organism descriptions
2. Add Sentence-BERT encoder
3. Create description dataset
4. Joint training on taxonomy + descriptions
5. Enable natural language queries

**Effort:** Medium (2-3 weeks)

---

## 🎯 Recommended Implementation Order

### Phase 1: Foundation (✅ DONE)
- [x] Clean repository structure
- [x] Fix data handling bugs
- [x] Validate training pipeline
- [x] Create universal visualization tool
- [x] Write comprehensive documentation

### Phase 2: Text Integration (NEXT - 2-3 weeks)
- [ ] **Species Names** (Priority 1)
  - Simplest extension
  - High user value (searchability)
  - Foundation for other text extensions
  
**Steps:**
1. Load names from `names.dmp`
2. Implement BERT encoder
3. Create joint training loop
4. Update visualization for text queries
5. Validate on text search tasks

### Phase 3: Multi-Modal (4-6 weeks)
- [ ] **Additional Features** (Priority 2)
  - Collect metadata
  - Medium complexity
  - Good for downstream tasks

- [ ] **Word Descriptions** (Priority 3)
  - Requires text infrastructure from Phase 2
  - Enables semantic search
  - Educational value

### Phase 4: Advanced (6-8 weeks)
- [ ] **Protein Embeddings** (Priority 4)
  - Most complex
  - Requires large compute
  - High biological value

## 📋 Implementation Template

For each extension, follow this template:

### File Structure
```
taxembed/
├── src/taxembed/
│   ├── encoders/              # NEW
│   │   ├── text_encoder.py
│   │   ├── protein_encoder.py
│   │   └── feature_encoder.py
│   ├── multimodal/            # NEW
│   │   ├── fusion.py
│   │   └── joint_training.py
│   └── models/
│       └── poincare_multimodal.py  # NEW
│
├── scripts/
│   ├── prepare_text_data.py   # NEW
│   └── train_multimodal.py    # NEW
│
└── tests/
    └── test_multimodal.py      # NEW
```

### Training Script
```python
# scripts/train_multimodal.py

import torch
from taxembed.models import MultiModalPoincare

def main():
    # Load data
    graph_data = load_graph("data/taxonomy_edges.mapped.edgelist")
    text_data = load_text("data/organism_names.txt")  # NEW
    
    # Model
    model = MultiModalPoincare(
        graph_dim=10,
        text_dim=768,  # BERT hidden size
        fusion_dim=10
    )
    
    # Training
    for epoch in range(epochs):
        # Graph loss (existing)
        graph_loss = train_graph_batch(model, graph_data)
        
        # Text loss (NEW)
        text_loss = train_text_batch(model, text_data)
        
        # Combined
        total_loss = graph_loss + lambda_text * text_loss
        total_loss.backward()
```

### Evaluation
```python
# Evaluate multimodal
def evaluate_multimodal(model):
    # Graph-based (existing)
    graph_metrics = evaluate_nearest_neighbors(model)
    
    # Text-based (NEW)
    text_metrics = evaluate_text_search(model)
    # e.g., "sapiens" → find Homo sapiens
    
    # Joint
    joint_metrics = evaluate_cross_modal(model)
    # e.g., TaxID → text, text → TaxID
    
    return {**graph_metrics, **text_metrics, **joint_metrics}
```

## 🔬 Research Questions

Each extension opens research opportunities:

### Names
- How much does text improve hierarchy learning?
- Can we handle multilingual names?
- How to deal with synonyms?

### Proteins
- Does protein sequence improve organism clustering?
- Can we predict protein function from taxonomy?
- How to handle horizontal gene transfer?

### Features
- Which features are most informative?
- Can we predict missing features?
- How to handle sparse features?

### Descriptions
- Can we generate organism descriptions?
- How to link to literature?
- Can we answer biological questions?

## 📊 Success Metrics

### Quantitative
- Nearest neighbor accuracy
- Cluster purity
- Text search recall@k
- Cross-modal retrieval accuracy

### Qualitative
- Biologically meaningful clusters
- Useful for downstream tasks
- Better interpretability
- User satisfaction

## 🎓 Publication Opportunities

Each extension could lead to publications:

1. **Multimodal Taxonomy Embeddings** - combine graph + text + features
2. **Hyperbolic Protein-Taxonomy Embeddings** - protein function in hyperbolic space
3. **Natural Language Queries for Organisms** - semantic search in taxonomy
4. **Hierarchical Biological Embeddings** - comprehensive benchmark

## 💡 Additional Ideas

### Short-term Enhancements
- [ ] Add more taxonomic groups to visualization
- [ ] Create web interface for exploration
- [ ] Add batch inference API
- [ ] Create pre-trained model checkpoints
- [ ] Add Docker container

### Long-term Vision
- [ ] Real-time taxonomy updates
- [ ] Interactive exploration tool
- [ ] Integration with biological databases
- [ ] API for downstream applications
- [ ] Educational platform

## 🤝 Getting Started

**Ready to implement extensions?**

1. Read `POINCARE_EMBEDDINGS_EXPLAINED.md` for technical background
2. Start with **Species Names** (easiest, high value)
3. Follow the implementation template
4. Add tests as you go
5. Update documentation
6. Create PR for review

**Questions?** See `CONTRIBUTING.md` or open an issue.

---

**Current focus:** Graph structure works excellently! Extensions should preserve this while adding complementary information.
