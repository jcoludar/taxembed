# Theory: Poincaré Embeddings for Hierarchies

Understanding the mathematics and intuition behind hyperbolic embeddings.

## The Problem: Representing Hierarchies

Traditional embeddings (Word2Vec, GloVe) use **Euclidean space** (flat space). But hierarchical data like taxonomies have **exponential growth**:

```
Root:     1 node
Level 1:  10 nodes
Level 2:  100 nodes
Level 3:  1,000 nodes
Level 4:  10,000 nodes
```

In flat Euclidean space, you need **exponentially growing dimensions** to represent this without distortion.

## The Solution: Hyperbolic Space

**Poincaré embeddings** use **hyperbolic geometry** where:

- Distance grows **exponentially** as you move from center to boundary
- Perfect for hierarchies: root near center, leaves near boundary
- Can represent exponential growth in **constant dimensions**

### The Poincaré Ball Model

The n-dimensional Poincaré ball is:

```
B^n = {x ∈ R^n : ||x|| < 1}
```

All points inside the unit ball. The boundary (||x|| = 1) is "at infinity".

### Poincaré Distance

The hyperbolic distance between points u and v is:

```
d(u,v) = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))
```

Key properties:

- Distances grow exponentially near boundary
- Center (origin) is "special" - represents root
- Angular distance represents similarity

## Why It Works for Taxonomies

### 1. Natural Hierarchy Encoding

In the Poincaré ball:

- **Root organisms** (cellular life) → center (||x|| ≈ 0.1)
- **Intermediate levels** (kingdoms, phyla) → middle (||x|| ≈ 0.5)
- **Leaf organisms** (species) → boundary (||x|| ≈ 0.95)

Depth in taxonomy directly corresponds to norm (distance from origin).

### 2. Exponential Capacity

Near the boundary, even small angular differences create large hyperbolic distances. This allows:

- Millions of species to fit in 10-20 dimensions
- Each level has exponentially more "room" than the previous

### 3. Hierarchy Preservation

The loss function encourages:

- Ancestors closer to descendants than to random nodes
- Deeper pairs (great-grandparent → descendant) to have larger distances
- Siblings (same depth) to have similar norms but different angles

## Training Methodology

### Loss Function

We use a **ranking loss with margin**:

```
L = max(0, d(ancestor, descendant) - d(ancestor, negative) + margin)
```

This encourages:

- `d(ancestor, descendant)` < `d(ancestor, negative) + margin`
- Ancestors should be closer to their descendants than to random nodes

### Radial Regularization

We add a penalty to encourage depth → radius mapping:

```
R = λ Σ (||embedding_i|| - target_radius_i)²
```

Where `target_radius = 0.1 + (depth/max_depth) × 0.85`

This ensures:

- Root nodes stay near center
- Leaf nodes move toward boundary
- Smooth gradient across depths

### Hard Negative Sampling

Instead of random negatives, we sample **cousins** (nodes at same depth):

- More informative: teaches model to distinguish siblings
- Depth-stratified: ensures coverage across all levels
- Harder: creates better separation

## Optimization Challenges

### Ball Constraint

All embeddings must satisfy ||x|| < 1. We enforce this with:

1. **Gradient clipping**: Prevents large jumps
2. **Selective projection**: After each batch, project updated embeddings
3. **Full projection**: At epoch end, ensure all embeddings inside ball

### Riemannian Optimization

Standard SGD assumes Euclidean space. For hyperbolic space, we could use:

- **Riemannian SGD**: Updates along geodesics
- **Exponential map**: Maps tangent vectors to manifold

Currently, we use standard SGD with careful projection - simpler and works well.

## Theoretical Guarantees

### Embedding Capacity

In d-dimensional Poincaré ball, the number of points with pairwise distance ≥ δ grows as:

```
N ≈ exp(d × δ)
```

For taxonomies:

- 10D can embed ~10^6 nodes with good separation
- 20D can embed ~10^12 nodes

### Distortion Bounds

For trees with branching factor b and depth h:

- **Euclidean**: needs O(b^h) dimensions
- **Hyperbolic**: needs O(h) dimensions

Exponential improvement!

## Comparison to Other Approaches

### vs. Euclidean Embeddings

- ❌ Euclidean: Poor for hierarchies, needs many dimensions
- ✅ Hyperbolic: Natural fit, constant dimensions

### vs. Graph Neural Networks

- ❌ GNNs: Complex, slow, need many layers for deep trees
- ✅ Hyperbolic: Direct embedding, fast training

### vs. Order Embeddings

- ❌ Order: Can represent partial orders, but wastes dimensions
- ✅ Hyperbolic: Efficient, captures similarity and order

## Practical Considerations

### Choosing Dimensionality

Rule of thumb:

- **Small taxonomies** (<10K nodes): 5-10D sufficient
- **Medium** (10K-1M): 10-20D
- **Large** (>1M): 20-50D

Higher dimensions allow more nuance but train slower.

### Choosing Regularization

λ controls radial constraint:

- **Too small** (λ < 0.01): Nodes may ignore depth structure
- **Just right** (λ = 0.1): Enforces depth → radius while allowing flexibility
- **Too large** (λ > 0.5): Over-constrained, poor performance

### Numerical Stability

Near the boundary (||x|| ≈ 1), distances can explode. We handle this with:

- Clamping norms: `||x|| < 0.999` (not exactly 1)
- Small epsilon in formulas: Prevents division by zero
- Gradient clipping: Avoids NaN/Inf

## Mathematical Foundations

### Hyperbolic Geometry

The Poincaré ball is one model of hyperbolic space (constant negative curvature). Other models:

- **Hyperboloid**: Upper sheet of hyperboloid
- **Klein**: Projective model
- **Upper half-space**: Complex plane with Im(z) > 0

All are isometric (same geometry, different coordinates).

### Geodesics

Shortest paths in Poincaré ball are:

- **Through origin**: Straight lines
- **Not through origin**: Circular arcs perpendicular to boundary

### Curvature

Poincaré ball has constant negative curvature κ = -1. This creates exponential growth in volume:

```
Vol(ball of radius r) ∝ exp(r)
```

This is why it's perfect for trees!

## Further Reading

### Papers

- [Poincaré Embeddings (Nickel & Kiela, 2017)](https://arxiv.org/abs/1705.08039)
- [Hyperbolic Neural Networks (Ganea et al., 2018)](https://arxiv.org/abs/1805.09112)
- [Learning Continuous Hierarchies (Sala et al., 2018)](https://arxiv.org/abs/1806.03417)

### Books

- Anderson, J. W. (2005). Hyperbolic Geometry (Springer Undergraduate Mathematics)
- Ratcliffe, J. G. (2006). Foundations of Hyperbolic Manifolds

### Implementations

- [geoopt](https://github.com/geoopt/geoopt) - Riemannian optimization in PyTorch
- [hyperlib](https://github.com/lateral/hyperlib) - Hyperbolic geometry utilities
