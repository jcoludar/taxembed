import numpy as np
from taxembed.eval.treedist import TreeDistance

# Hand tree (node id : parent):  0=root
#   0
#   ├─1        ├─2
#   │ ├─3      │
#   │ │ └─5    │
#   │ └─4      │
# parent[root] = root (self-loop convention)
PARENT = np.array([0, 0, 0, 1, 1, 3])
DEPTH  = np.array([0, 1, 1, 2, 2, 3])


def _td():
    return TreeDistance(PARENT, DEPTH)


def test_lca_basic():
    td = _td()
    a = np.array([3, 5, 5, 1, 4])
    b = np.array([4, 4, 2, 1, 5])
    assert list(td.lca(a, b)) == [1, 1, 0, 1, 1]


def test_path_length():
    td = _td()
    a = np.array([3, 5, 5, 1])
    b = np.array([4, 4, 2, 1])
    # d(3,4)=2; d(5,4)=3; d(5,2)=4; d(1,1)=0
    assert list(td.path_length(a, b)) == [2, 3, 4, 0]


def test_lca_depth_distance():
    td = _td()
    a = np.array([5, 5])
    b = np.array([4, 2])
    # relatedness distance = (depth[a]-depth[lca]) ... we test the shallow-split form:
    # n_levels_to_split = depth[a] + depth[b] - 2*depth[lca] is path_length; lca_depth returns depth[lca]
    assert list(td.lca_depth(a, b)) == [1, 0]
