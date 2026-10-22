"""Immutable source dependency facts; no execution or workflow state."""
from dataclasses import dataclass
from graphlib import TopologicalSorter



@dataclass(frozen=True, slots=True)
class Dependency:
    node_id: str
    event: str
    delay_ns: int | None = None

    def __post_init__(self):
        if not self.node_id or self.event not in {"started", "completed"}:
            raise ValueError("dependency requires a node and started/completed event")
        if self.delay_ns is not None and (type(self.delay_ns) is not int or self.delay_ns < 0):
            raise ValueError("dependency interval must be nonnegative or unknown")


@dataclass(frozen=True, slots=True)
class CausalNode:
    graph_id: str
    node_id: str
    branch_id: str
    root_ns: int
    not_before_ns: int
    dependencies: tuple[Dependency, ...] = ()

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in (self.graph_id, self.node_id, self.branch_id)):
            raise ValueError("causal identities must not be empty")
        for name in ("root_ns", "not_before_ns"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.not_before_ns < self.root_ns:
            raise ValueError("node precedes its root arrival")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("dependencies must be immutable")

    @classmethod
    def decode(cls, value):
        if value is None:
            return None
        return cls(value["graph_id"], value["node_id"], value["branch_id"], value["root_ns"],
                   value["not_before_ns"], tuple(Dependency(**edge) for edge in value["dependencies"]))


def validate_graph(nodes):
    by_id, branches, edges, roots = {}, {}, {}, {}
    for node in nodes:
        if node.node_id in by_id:
            raise ValueError("duplicate causal node identity")
        if roots.setdefault(node.graph_id, node.root_ns) != node.root_ns:
            raise ValueError("inconsistent root arrival within a graph")
        by_id[node.node_id] = node
        branches.setdefault((node.graph_id, node.branch_id), []).append(node)
        edges[node.node_id] = {edge.node_id for edge in node.dependencies}
    for node in by_id.values():
        for parent_id in edges[node.node_id]:
            parent = by_id.get(parent_id)
            if parent is None or parent.graph_id != node.graph_id:
                raise ValueError("dependency references a missing node or another graph")
    for branch in branches.values():
        for previous, current in zip(branch, branch[1:]):
            if current.not_before_ns < previous.not_before_ns:
                raise ValueError("causal branch order is inconsistent")
            edges[current.node_id].add(previous.node_id)
    tuple(TopologicalSorter(edges).static_order())
    widths = {}
    for graph, _ in branches:
        widths[graph] = widths.get(graph, 0) + 1
    return max(widths.values(), default=0)
