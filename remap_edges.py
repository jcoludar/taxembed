import sys, collections

in_path = sys.argv[1]
out_edges = in_path.replace(".edgelist", ".mapped.edgelist")
out_map = in_path.replace(".edgelist", ".mapping.tsv")

nodes = collections.OrderedDict()  # preserves order of first appearance
edges = []

with open(in_path) as f:
    for ln, line in enumerate(f, 1):
        line=line.strip()
        if not line: 
            continue
        parts = line.split()
        if len(parts)!=2:
            raise ValueError(f"Bad line {ln}: {line!r}")
        u, v = parts
        # register nodes
        if u not in nodes: nodes[u] = len(nodes)
        if v not in nodes: nodes[v] = len(nodes)
        edges.append((nodes[u], nodes[v]))

with open(out_edges, "w") as fo:
    for u, v in edges:
        fo.write(f"{u} {v}\n")

with open(out_map, "w") as fm:
    fm.write("taxid\tidx\n")
    for taxid, idx in nodes.items():
        fm.write(f"{taxid}\t{idx}\n")

print(f"Wrote: {out_edges} and {out_map}. Nodes: {len(nodes)}, edges: {len(edges)}")
