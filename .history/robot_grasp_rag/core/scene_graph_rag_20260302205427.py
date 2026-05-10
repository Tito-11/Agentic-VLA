"""Core module implementation."""

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class SpatialRelation(Enum):


    """SpatialRelation class."""
    # Note

    ON_TOP_OF = "on_top_of"         # Note
    UNDER = "under"                 # Note
    INSIDE = "inside"               # Note
    CONTAINS = "contains"           # Note
    
    # Note

    NEXT_TO = "next_to"             # Note
    LEFT_OF = "left_of"             # Note
    RIGHT_OF = "right_of"           # Note
    IN_FRONT_OF = "in_front_of"     # Note
    BEHIND = "behind"               # Note
    
    # Note

    OCCLUDES = "occludes"           # Note
    OCCLUDED_BY = "occluded_by"     # Note
    
    # Note

    SUPPORTS = "supports"           # Note
    SUPPORTED_BY = "supported_by"   # Note
    
    # Note

    NEAR = "near"                   # Note
    FAR = "far"                     # Note


class ObjectState(Enum):


    """ObjectState class."""
    EMPTY = "empty"                 # Note
    FILLED = "filled"               # Note
    OPEN = "open"                   # Note
    CLOSED = "closed"               # Note
    STABLE = "stable"               # Note
    UNSTABLE = "unstable"           # Note


@dataclass
class SceneObject:
    """SceneObject class."""
    id: str
    name: str
    category: str
    
    # Note

    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    bbox_3d: Optional[Dict[str, float]] = None  # min_x, max_x, min_y, max_y, min_z, max_z
    
    # Note

    affordance_type: Optional[str] = None
    material: Optional[str] = None
    is_fragile: bool = False
    
    # Note

    state: ObjectState = ObjectState.STABLE
    contains: Optional[str] = None  # Note
    
    # Note

    is_graspable: bool = True
    is_target: bool = False  # Note


@dataclass
class SceneRelation:
    """SceneRelation class."""
    subject_id: str           # Note
    relation: SpatialRelation # Note
    object_id: str            # Note
    confidence: float = 1.0   # Note
    
    # Note

    distance: Optional[float] = None  # Note
    direction: Optional[Tuple[float, float, float]] = None  # Note


@dataclass
class GraspConstraint:
    """GraspConstraint class."""
    constraint_type: str      # "collision", "support", "content", "occlusion"
    source_object: str        # Note
    description: str          # Note
    severity: str             # "critical", "warning", "info"
    suggested_action: str     # Note


class SceneGraph:


    """SceneGraph class."""
    def __init__(self):
        self.objects: Dict[str, SceneObject] = {}
        self.relations: List[SceneRelation] = []
        
        # Note

        self._subject_index: Dict[str, List[SceneRelation]] = defaultdict(list)
        self._object_index: Dict[str, List[SceneRelation]] = defaultdict(list)
        self._relation_index: Dict[SpatialRelation, List[SceneRelation]] = defaultdict(list)
    
    def add_object(self, obj: SceneObject):
    
        """add_object function."""
        self.objects[obj.id] = obj
    
    def add_relation(self, relation: SceneRelation):
    
        """add_relation function."""
        self.relations.append(relation)
        self._subject_index[relation.subject_id].append(relation)
        self._object_index[relation.object_id].append(relation)
        self._relation_index[relation.relation].append(relation)
    
    def get_neighbors(self, obj_id: str, relation_type: Optional[SpatialRelation] = None) -> List[str]:
    
        """get_neighbors function."""
        neighbors = set()
        
        for rel in self._subject_index[obj_id]:
            if relation_type is None or rel.relation == relation_type:
                neighbors.add(rel.object_id)
        
        for rel in self._object_index[obj_id]:
            if relation_type is None or rel.relation == relation_type:
                neighbors.add(rel.subject_id)
        
        return list(neighbors)
    
    def get_relations_for(self, obj_id: str) -> List[SceneRelation]:
    
        """get_relations_for function."""
        relations = []
        relations.extend(self._subject_index[obj_id])
        relations.extend(self._object_index[obj_id])
        return relations
    
    def get_path(self, from_id: str, to_id: str, max_depth: int = 5) -> Optional[List[str]]:
    
        """get_path function."""
        if from_id == to_id:
            return [from_id]
        
        visited = set()
        queue = [(from_id, [from_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor == to_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
    
        """to_dict function."""
        return {
            "objects": {k: {
                "name": v.name,
                "category": v.category,
                "position": v.position,
                "affordance_type": v.affordance_type,
                "state": v.state.value if v.state else None,
            } for k, v in self.objects.items()},
            "relations": [
                {
                    "subject": r.subject_id,
                    "relation": r.relation.value,
                    "object": r.object_id,
                    "confidence": r.confidence,
                }
                for r in self.relations
            ],
        }


class SceneGraphBuilder:


    """SceneGraphBuilder class."""
    SCENE_ANALYSIS_PROMPT = """Analyze the objects in the scene and their spatial relationships。

Output in JSON format:
```json
{
    "objects": [
        {
            "id": "obj_1",
            "name": "coffee cup",
            "category": "cup",
            "position": {"x": 0.3, "y": 0.2, "z": 0.1},
            "state": "filled",
            "contains": "water"
        }
    ],
    "relations": [
        {
            "subject": "obj_1",
            "relation": "on_top_of",
            "object": "obj_2"
        }
    ]
}
```

Supported relation types: on_top_of, under, inside, contains, next_to, left_of, right_of, in_front_of, behind, occludes, supports, near
Object states: empty, filled, open, closed, stable, unstable
"""
    
    def __init__(self, vlm_client = None):
        self.vlm_client = vlm_client
    
    def build_from_vlm(self, image, target_object: Optional[str] = None) -> SceneGraph:
    
        """build_from_vlm function."""
        if self.vlm_client is None:
            return self._build_mock_scene_graph(target_object)
        
        # Note

        response = self.vlm_client.generate(
            image=image,
            prompt=self.SCENE_ANALYSIS_PROMPT,
            max_tokens=1024,
        )
        
        return self._parse_vlm_response(response, target_object)
    
    def _parse_vlm_response(self, response: str, target_object: Optional[str]) -> SceneGraph:
    
        """_parse_vlm_response function."""
        import re
        
        graph = SceneGraph()
        
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)
            
            # Note

            for obj_data in data.get("objects", []):
                obj = SceneObject(
                    id=obj_data.get("id", f"obj_{len(graph.objects)}"),
                    name=obj_data.get("name", "unknown"),
                    category=obj_data.get("category", "unknown"),
                    position=obj_data.get("position", {"x": 0, "y": 0, "z": 0}),
                    state=ObjectState(obj_data.get("state", "stable")),
                    contains=obj_data.get("contains"),
                )
                
                if target_object and target_object.lower() in obj.name.lower():
                    obj.is_target = True
                
                graph.add_object(obj)
            
            # Note

            for rel_data in data.get("relations", []):
                try:
                    rel = SceneRelation(
                        subject_id=rel_data["subject"],
                        relation=SpatialRelation(rel_data["relation"]),
                        object_id=rel_data["object"],
                    )
                    graph.add_relation(rel)
                except:
                    continue
                    
        except Exception as e:
            print(f"[SceneGraphBuilder] Build failed: {e}")
        
        return graph
    
    def _build_mock_scene_graph(self, target_object: Optional[str]) -> SceneGraph:
    
        """_build_mock_scene_graph function."""
        graph = SceneGraph()
        
        # Note

        table = SceneObject(
            id="table_1",
            name="dining table",
            category="table",
            position={"x": 0, "y": 0, "z": 0},
        )
        
        cup = SceneObject(
            id="cup_1",
            name="coffee cup",
            category="cup",
            position={"x": 0.2, "y": 0.1, "z": 0.1},
            is_fragile=True,
            state=ObjectState.FILLED,
            contains="water",
            is_target=(target_object and "cup" in target_object.lower()),
        )
        
        book = SceneObject(
            id="book_1",
            name="notebook",
            category="book",
            position={"x": 0.35, "y": 0.1, "z": 0.05},
        )
        
        graph.add_object(table)
        graph.add_object(cup)
        graph.add_object(book)
        
        # Note

        graph.add_relation(SceneRelation("cup_1", SpatialRelation.ON_TOP_OF, "table_1"))
        graph.add_relation(SceneRelation("book_1", SpatialRelation.ON_TOP_OF, "table_1"))
        graph.add_relation(SceneRelation("cup_1", SpatialRelation.NEXT_TO, "book_1", distance=0.15))
        graph.add_relation(SceneRelation("cup_1", SpatialRelation.CONTAINS, "water_1"))
        
        return graph


class ConstraintAnalyzer:


    """ConstraintAnalyzer class."""
    def analyze(self, graph: SceneGraph, target_id: str) -> List[GraspConstraint]:
    
        """analyze function."""
        constraints = []
        target = graph.objects.get(target_id)
        
        if target is None:
            return constraints
        
        # Note

        if target.state == ObjectState.FILLED and target.contains:
            constraints.append(GraspConstraint(
                constraint_type="content",
                source_object=target.contains,
                description=f"Contains {target.contains}, handle with care",
                severity="critical",
                suggested_action="grasp_keep_level",
            ))
        
        # Note

        for rel in graph.get_relations_for(target_id):
            if rel.relation == SpatialRelation.SUPPORTS and rel.subject_id == target_id:
                # Note

                supported_obj = graph.objects.get(rel.object_id)
                if supported_obj:
                    constraints.append(GraspConstraint(
                        constraint_type="support",
                        source_object=rel.object_id,
                        description=f"Supports {supported_obj.name}, clear before grasping",
                        severity="critical",
                        suggested_action="remove_supported_first",
                    ))
        
        # Note

        for rel in graph.get_relations_for(target_id):
            if rel.relation == SpatialRelation.OCCLUDED_BY and rel.subject_id == target_id:
                occluder = graph.objects.get(rel.object_id)
                if occluder:
                    constraints.append(GraspConstraint(
                        constraint_type="occlusion",
                        source_object=rel.object_id,
                        description=f"Occluded by {occluder.name}, adjust approach",
                        severity="warning",
                        suggested_action="approach_from_side",
                    ))
        
        # Note

        neighbors = graph.get_neighbors(target_id, SpatialRelation.NEXT_TO)
        for neighbor_id in neighbors:
            neighbor = graph.objects.get(neighbor_id)
            if neighbor and neighbor.is_fragile:
                # Note

                for rel in graph._subject_index[target_id]:
                    if rel.object_id == neighbor_id and rel.distance:
                        if rel.distance < 0.1: # Note
                            constraints.append(GraspConstraint(
                                constraint_type="collision",
                                source_object=neighbor_id,
                                description=f"Near {neighbor.name} (distance {rel.distance*100:.0f}cm)",
                                severity="warning",
                                suggested_action="careful_approach",
                            ))
        
        return constraints


class SceneGraphRetriever:


    """SceneGraphRetriever class."""
    def __init__(self):
        # Note

        self.scene_experiences: Dict[str, Dict[str, Any]] = {}
        
        # Note

        self.graph_builder = SceneGraphBuilder()
        
        # Note

        self.constraint_analyzer = ConstraintAnalyzer()
    
    def add_experience(
        self,
        experience_id: str,
        scene_graph: SceneGraph,
        target_object_id: str,
        grasp_result: Dict[str, Any],
    ):
        """add_experience function."""
        self.scene_experiences[experience_id] = {
            "scene_graph": scene_graph,
            "target_id": target_object_id,
            "grasp_result": grasp_result,
            "constraints": self.constraint_analyzer.analyze(scene_graph, target_object_id),
        }
    
    def compute_scene_similarity(self, graph1: SceneGraph, graph2: SceneGraph) -> float:
    
        """compute_scene_similarity function."""
        score = 0.0
        
        # Note

        categories1 = set(obj.category for obj in graph1.objects.values())
        categories2 = set(obj.category for obj in graph2.objects.values())
        
        if categories1 and categories2:
            category_overlap = len(categories1 & categories2) / len(categories1 | categories2)
            score += 0.4 * category_overlap
        
        # Note

        relations1 = set(r.relation for r in graph1.relations)
        relations2 = set(r.relation for r in graph2.relations)
        
        if relations1 and relations2:
            relation_overlap = len(relations1 & relations2) / len(relations1 | relations2)
            score += 0.3 * relation_overlap
        
        # Note

        targets1 = [obj for obj in graph1.objects.values() if obj.is_target]
        targets2 = [obj for obj in graph2.objects.values() if obj.is_target]
        
        if targets1 and targets2:
            t1, t2 = targets1[0], targets2[0]
            if t1.category == t2.category:
                score += 0.2
            if t1.state == t2.state:
                score += 0.1
        
        return score
    
    def retrieve_similar_scenes(
        self,
        query_graph: SceneGraph,
        target_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float, List[GraspConstraint]]]:
        """retrieve_similar_scenes function."""
        results = []
        
        for exp_id, exp_data in self.scene_experiences.items():
            similarity = self.compute_scene_similarity(query_graph, exp_data["scene_graph"])
            results.append((exp_id, similarity, exp_data["constraints"]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class SceneGraphRAG:


    """SceneGraphRAG class."""
    def __init__(
        self,
        vlm_client=None,
        base_retriever=None,  # Note
    ):
        self.vlm_client = vlm_client
        self.base_retriever = base_retriever
        
        self.graph_builder = SceneGraphBuilder(vlm_client)
        self.constraint_analyzer = ConstraintAnalyzer()
        self.scene_retriever = SceneGraphRetriever()
    
    def retrieve_and_build_context(
        self,
        query_image,
        query_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """retrieve_and_build_context function."""
        # Note

        scene_graph = self.graph_builder.build_from_vlm(query_image, query_text)
        
        # Note

        target_id = None
        for obj_id, obj in scene_graph.objects.items():
            if obj.is_target:
                target_id = obj_id
                break
        
        # Note

        constraints = []
        if target_id:
            constraints = self.constraint_analyzer.analyze(scene_graph, target_id)
        
        # Note

        base_results = []
        if self.base_retriever:
            if hasattr(self.base_retriever, 'retrieve_and_build_context'):
                base_context = self.base_retriever.retrieve_and_build_context(
                    query_image=query_image,
                    query_text=query_text,
                    top_k=top_k,
                )
                base_results = base_context.get("experiences", [])
        
        # Note

        scene_results = self.scene_retriever.retrieve_similar_scenes(
            scene_graph, target_id or "", top_k=3
        )
        
        # Note

        context = {
            "query_text": query_text,
            "scene_graph": scene_graph.to_dict(),
            "target_object": scene_graph.objects.get(target_id).__dict__ if target_id else None,
            "constraints": [
                {
                    "type": c.constraint_type,
                    "source": c.source_object,
                    "description": c.description,
                    "severity": c.severity,
                    "action": c.suggested_action,
                }
                for c in constraints
            ],
            "experiences": base_results,
            "similar_scenes": [
                {"id": r[0], "similarity": r[1]}
                for r in scene_results
            ],
        }
        
        # Note

        context["scene_aware_prompt"] = self._generate_scene_prompt(
            scene_graph, target_id, constraints
        )
        
        return context
    
    def _generate_scene_prompt(
        self,
        graph: SceneGraph,
        target_id: Optional[str],
        constraints: List[GraspConstraint],
    ) -> str:
        """_generate_scene_prompt function."""
        lines = ["## Scene Context\n"]
        
        # Note

        if target_id and target_id in graph.objects:
            target = graph.objects[target_id]
            lines.append(f"### Target Object\n- Name: {target.name}\n- Category: {target.category}")
            if target.state != ObjectState.STABLE:
                lines.append(f"- State: {target.state.value}")
            if target.contains:
                lines.append(f"- Contains: {target.contains}")
        
        # Note

        if target_id:
            neighbors = graph.get_neighbors(target_id)
            if neighbors:
                lines.append("\n### objects")
                for n_id in neighbors[:5]:
                    if n_id in graph.objects:
                        n = graph.objects[n_id]
                        rels = [r.relation.value for r in graph.get_relations_for(n_id) 
                                if r.subject_id == target_id or r.object_id == target_id]
                        lines.append(f"- {n.name} ({', '.join(rels)})")
        
        # Note

        if constraints:
            lines.append("\n### ⚠️ Grasp Constraints")
            for c in constraints:
                icon = "🔴" if c.severity == "critical" else "🟡"
                lines.append(f"{icon} [{c.constraint_type}] {c.description}")
                lines.append(f"   → Suggestion: {c.suggested_action}")
        
        return "\n".join(lines)


# Note

def create_scene_graph_rag(
    vlm_client=None,
    base_retriever=None,
) -> SceneGraphRAG:
    """create_scene_graph_rag function."""
    return SceneGraphRAG(vlm_client, base_retriever)
