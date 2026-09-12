"""Frozen semantic schema for universal RoboTwin event pretraining.

The shared model sees no joint indices and no embodiment id.  Every task is
represented as a small typed scene graph plus a goal-relation graph.  The
``attribute`` names below are public RoboTwin task attributes, not learned
labels.  Adding a task only requires another declarative row when its objects
are already exposed by the simulator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


FORMAT = "etsf_robotwin2_universal_event_schema_v1"
MAX_NODES = 8
NODE_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 12
CANONICAL_ACTION_DIM = 14

NODE_TYPES = (
    "object",
    "container",
    "support",
    "articulated",
    "tool",
    "left_gripper",
    "right_gripper",
    "goal",
)
RELATION_NAMES = (
    "near",
    "held_by",
    "supported_by",
    "inside",
    "on_top_of",
    "open",
    "pressed",
    "activated",
    "released",
    "dropped",
    "lifted",
    "away_from",
)
EVENT_NAMES = (
    "approach",
    "grasp",
    "transport",
    "place",
    "release",
    "drop",
    "regrasp",
    "open",
    "activate",
    "idle",
)


@dataclasses.dataclass(frozen=True)
class NodeSpec:
    name: str
    node_type: str
    attribute: str
    movable: bool = False
    articulation: bool = False
    position_mode: str = "actor"
    functional_point_id: int = 0

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {self.node_type}")
        if self.position_mode not in {"actor", "functional_point", "actor_functional_midpoint"}:
            raise ValueError(f"unknown node position mode: {self.position_mode}")
        if self.functional_point_id < 0:
            raise ValueError("functional point id must be non-negative")


@dataclasses.dataclass(frozen=True)
class GoalEdge:
    relation: str
    source: str
    target: str
    value: int = 1

    def __post_init__(self) -> None:
        if self.relation not in RELATION_NAMES or self.value not in (0, 1):
            raise ValueError("invalid goal edge")


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task: str
    instruction: str
    nodes: tuple[NodeSpec, ...]
    goals: tuple[GoalEdge, ...]
    moving: str | None = None
    target: str | None = None
    xy_threshold_m: float = 0.06
    z_threshold_m: float = 0.04
    open_fraction: float = 0.4
    activation_attribute: str | None = None

    def __post_init__(self) -> None:
        names = {node.name for node in self.nodes}
        if len(names) != len(self.nodes) or len(self.nodes) + 2 > MAX_NODES:
            raise ValueError(f"invalid node roster for {self.task}")
        for goal in self.goals:
            if goal.source not in names or goal.target not in names:
                raise ValueError(f"goal references an unknown node in {self.task}")


def _n(name: str, node_type: str, attribute: str, **kwargs: Any) -> NodeSpec:
    return NodeSpec(name, node_type, attribute, **kwargs)


def _g(relation: str, source: str, target: str, value: int = 1) -> GoalEdge:
    return GoalEdge(relation, source, target, value)


# These tasks deliberately span placement, containment, stacking, articulation,
# contact activation and handover.  All five RoboTwin embodiments can execute
# the same public task class; robot-specific joints stay outside this schema.
TASK_SPECS: dict[str, TaskSpec] = {
    "move_can_pot": TaskSpec(
        "move_can_pot", "Move the can next to the pot.",
        (_n("moving", "object", "can", movable=True), _n("target", "support", "pot")),
        (_g("near", "moving", "target"), _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.20,
    ),
    "place_container_plate": TaskSpec(
        "place_container_plate", "Put the container on the plate.",
        (_n("moving", "object", "container", movable=True), _n("target", "support", "plate")),
        (_g("inside", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.05, z_threshold_m=0.04,
    ),
    "place_can_basket": TaskSpec(
        "place_can_basket", "Put the can in the basket.",
        (_n("moving", "object", "can", movable=True), _n("target", "container", "basket", movable=True)),
        (_g("inside", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.15, z_threshold_m=0.12,
    ),
    "place_empty_cup": TaskSpec(
        "place_empty_cup", "Put the empty cup on the coaster.",
        (_n("moving", "object", "cup", movable=True), _n("target", "support", "coaster")),
        (_g("on_top_of", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.035, z_threshold_m=0.03,
    ),
    "put_object_cabinet": TaskSpec(
        "put_object_cabinet", "Put the object into the cabinet.",
        (_n("moving", "object", "object", movable=True),
         _n("target", "container", "cabinet", articulation=True,
            position_mode="functional_point")),
        (_g("inside", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.07, z_threshold_m=0.14,
    ),
    "stack_blocks_two": TaskSpec(
        "stack_blocks_two", "Stack the second block on the first block.",
        (_n("target", "support", "block1", movable=True),
         _n("moving", "object", "block2", movable=True)),
        (_g("on_top_of", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.035, z_threshold_m=0.07,
    ),
    "hanging_mug": TaskSpec(
        "hanging_mug", "Hang the mug on the rack.",
        (_n("moving", "object", "mug", movable=True, position_mode="functional_point"),
         _n("target", "support", "rack", position_mode="actor_functional_midpoint")),
        (_g("supported_by", "moving", "target"), _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.05, z_threshold_m=0.16,
    ),
    "open_laptop": TaskSpec(
        "open_laptop", "Open the laptop lid.",
        (_n("articulated", "articulated", "laptop", articulation=True),),
        (_g("open", "articulated", "articulated"),), open_fraction=0.4,
    ),
    "open_microwave": TaskSpec(
        "open_microwave", "Open the microwave door.",
        (_n("articulated", "articulated", "microwave", articulation=True),),
        (_g("open", "articulated", "articulated"),), open_fraction=0.6,
    ),
    "press_stapler": TaskSpec(
        "press_stapler", "Press the stapler.",
        (_n("tool", "tool", "stapler"),),
        (_g("pressed", "tool", "tool"),), activation_attribute="stage_success_tag",
    ),
    "click_bell": TaskSpec(
        "click_bell", "Press the bell.",
        (_n("tool", "tool", "bell"),),
        (_g("activated", "tool", "tool"),), activation_attribute="stage_success_tag",
    ),
    "handover_block": TaskSpec(
        "handover_block", "Hand over the block and place it on the target.",
        (_n("moving", "object", "box", movable=True, position_mode="functional_point"),
         _n("target", "support", "target_box", position_mode="functional_point",
            functional_point_id=1)),
        (_g("on_top_of", "moving", "target"), _g("supported_by", "moving", "target"),
         _g("released", "moving", "moving")),
        "moving", "target", xy_threshold_m=0.04, z_threshold_m=0.03,
    ),
}


def task_spec_payload() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "max_nodes": MAX_NODES,
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "canonical_action_dim": CANONICAL_ACTION_DIM,
        "node_types": list(NODE_TYPES),
        "relations": list(RELATION_NAMES),
        "events": list(EVENT_NAMES),
        "tasks": {name: dataclasses.asdict(spec) for name, spec in TASK_SPECS.items()},
    }


def schema_sha256() -> str:
    raw = json.dumps(task_spec_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
