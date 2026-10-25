"""A quick tidy for node groups the add-on builds: columns by link depth."""

from collections import defaultdict

COLUMN_GAP, ROW_GAP = 60.0, 30.0
SOCKET_HEIGHT, HEADER_HEIGHT = 22.0, 40.0  # node dimensions are zero without a UI


def _height(node):
    sockets = sum(s.enabled and not s.hide for s in (*node.inputs, *node.outputs))
    return HEADER_HEIGHT + SOCKET_HEIGHT * sockets + (60.0 if node.bl_idname in ("ShaderNodeValToRGB", "ShaderNodeFloatCurve") else 0.0)


def arrange(tree):
    """Place each node one column right of its furthest-left input, stacking
    each column top-down in its old vertical order. Frames are left alone.
    """
    nodes = [n for n in tree.nodes if n.bl_idname != "NodeFrame"]
    inputs = defaultdict(set)
    for link in tree.links:
        if link.from_node != link.to_node:
            inputs[link.to_node].add(link.from_node)
    depth = {}

    def column(node, seen=()):
        if node not in depth:
            parents = [p for p in inputs[node] if p not in seen]
            depth[node] = 1 + max((column(p, (*seen, node)) for p in parents), default=-1)
        return depth[node]

    columns = defaultdict(list)
    for node in nodes:
        columns[column(node)].append(node)
    x = 0.0
    for index in sorted(columns):
        stack = sorted(columns[index], key=lambda n: -n.location.y)
        y = 0.0
        for node in stack:
            node.location = (x, y)
            y -= _height(node) + ROW_GAP
        x += max(n.width for n in stack) + COLUMN_GAP
