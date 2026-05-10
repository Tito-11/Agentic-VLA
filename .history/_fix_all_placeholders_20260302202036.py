"""
Comprehensive fix for all remaining placeholder strings ('sample', 'General adjustment', 
'General factor', 'EN' fragments) across robot_grasp_rag Python files.
Uses exact string replacement with single-occurrence matching for safety.
"""
import os
import re

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"
total_changes = 0

def fix(rel_path, pairs):
    """Apply list of (old, new) exact string replacements to a file. Each pair replaces first occurrence only."""
    global total_changes
    fp = os.path.join(ROOT, rel_path.replace("/", os.sep))
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    orig = content
    count = 0
    for old, new in pairs:
        if old in content:
            content = content.replace(old, new, 1)
            count += 1
        else:
            print(f"  MISS in {rel_path}: {old[:80]!r}")
    if content != orig:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  OK {rel_path}: {count} replacements")
        total_changes += count
    else:
        print(f"  SKIP {rel_path}: no changes")


# =============================================================================
# 1. scripts/build_knowledge_base.py
# =============================================================================
print("\n=== build_knowledge_base.py ===")
fix("scripts/build_knowledge_base.py", [
    # Cup 1
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample",',
     '"object_name": "red ceramic mug",\n            "category": "cup",\n            "description": "Standard ceramic mug with handle",'),
    # Cup 2
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample, sample",',
     '"object_name": "large coffee cup",\n            "category": "cup",\n            "description": "Large white coffee cup, wide mouth",'),
    # Cup 3 + key_insight
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.10},',
     '"object_name": "glass water cup",\n            "category": "cup",\n            "description": "Transparent glass cup, fragile",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.10},'),
    ('"key_insight": "sample, sample",\n            "color": "transparent",',
     '"key_insight": "Fragile glass, use gentle grip",\n            "color": "transparent",'),
    # Tool 1
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.15, "z": 0.05},',
     '"object_name": "screwdriver",\n            "category": "tool",\n            "description": "Phillips head screwdriver with rubber grip",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.15, "z": 0.05},'),
    ('"key_insight": "grasp sample",\n            "color": "red",',
     '"key_insight": "Grasp the handle firmly",\n            "color": "red",'),
    # Tool 2
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.04},',
     '"object_name": "adjustable wrench",\n            "category": "tool",\n            "description": "8-inch adjustable wrench",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.04},'),
    ('"key_insight": "grasp sample",\n            "color": "blue",',
     '"key_insight": "Grip the handle, avoid the jaw",\n            "color": "blue",'),
    # Tool 3
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": 0.05, "z": 0.03},',
     '"object_name": "needle-nose pliers",\n            "category": "tool",\n            "description": "Needle-nose pliers with insulated handle",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": 0.05, "z": 0.03},'),
    ('"key_insight": "grasp sample",\n            "color": "silver",',
     '"key_insight": "Firm grip on handle section",\n            "color": "silver",'),
    # Container 1
    ('"object_name": "sample",\n            "category": "container",\n            "description": "500ml sample",',
     '"object_name": "water bottle",\n            "category": "container",\n            "description": "500ml plastic water bottle",'),
    # Container 2
    ('"object_name": "sample",\n            "category": "container",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.12, "z": 0.06},',
     '"object_name": "metal thermos",\n            "category": "container",\n            "description": "Stainless steel thermos with lid",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.12, "z": 0.06},'),
    ('"key_insight": "sample, sample",\n            "color": "silver",',
     '"key_insight": "Heavy container, firm wrap grip",\n            "color": "silver",'),
    # Food 1
    ('"object_name": "sample",\n            "category": "food",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.04},',
     '"object_name": "apple",\n            "category": "food",\n            "description": "Red apple, round and smooth",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.04},'),
    # Food 2
    ('"object_name": "sample",\n            "category": "food",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.03},',
     '"object_name": "banana",\n            "category": "food",\n            "description": "Yellow banana, curved shape",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.03},'),
    # Electronics 1
    ('"object_name": "sample",\n            "category": "electronics",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.08, "z": 0.01},',
     '"object_name": "smartphone",\n            "category": "electronics",\n            "description": "Flat smartphone with glass screen",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.08, "z": 0.01},'),
    # Electronics 2
    ('"object_name": "sample",\n            "category": "electronics",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.02},',
     '"object_name": "wireless mouse",\n            "category": "electronics",\n            "description": "Ergonomic wireless mouse",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.02},'),
    # Print statements
    ('print(f"\\n[INFO] sample {len(samples)} experiences")',
     'print(f"\\n[INFO] Processing {len(samples)} experiences")'),
    ('print(f"\\n[{i+1}/{len(samples)}] sample: {sample[\'object_name\']}")',
     'print(f"\\n[{i+1}/{len(samples)}] Adding: {sample[\'object_name\']}")'),
    ('print(f"- sample: {count}")',
     'print(f"- Total entries: {count}")'),
    ('print(f"[INFO] sample {json_path} sample {len(data)} experiences")',
     'print(f"[INFO] Imported from {json_path}, {len(data)} experiences")'),
    ('help="sample"', 'help="Collection name"'),
    # Remaining mixed placeholders
    ('f"- imagepath: {image_dir}"', 'f"- Image path: {image_dir}"'),
    ('f"knowledge_baseENdone!"', 'f"Knowledge base build complete!"'),
    ('help="outputEN"', 'help="Output directory"'),
    ('help="images"', 'help="Skip image generation"'),
])


# =============================================================================
# 2. knowledge_base/schema.py
# =============================================================================
print("\n=== schema.py ===")
fix("knowledge_base/schema.py", [
    # Position3D
    ('description="X sample")\n    y: float = Field(..., description="Y sample")\n    z: float = Field(..., description="Z sample")',
     'description="X coordinate (meters)")\n    y: float = Field(..., description="Y coordinate (meters)")\n    z: float = Field(..., description="Z coordinate (meters)")'),
    # Quaternion
    ('description="X sample")\n    qy: float = Field(0.0, description="Y sample")\n    qz: float = Field(0.0, description="Z sample")\n    qw: float = Field(1.0, description="W sample")',
     'description="X component")\n    qy: float = Field(0.0, description="Y component")\n    qz: float = Field(0.0, description="Z component")\n    qw: float = Field(1.0, description="W component (scalar)")'),
    # GraspPose
    ('description="grasp sampleposition"', 'description="Grasp target position"'),
    ('description="grasp_pose"', 'description="Grasp orientation (quaternion)"'),
    ('description="gripper_width"', 'description="Gripper opening width (meters)"'),
    # ObjectInfo
    ('description="objectEN"', 'description="Object name"'),
    ('description="objectclassEN（sample cup, tool, container）"', 'description="Object category (e.g. cup, tool, container)"'),
    ('description="objectdescription"', 'description="Object description"'),
    ('description="sample（sample glass, metal, plastic）"', 'description="Material type (e.g. glass, metal, plastic)"'),
    ('weight_kg: Optional[float] = Field(None, description="sample")', 'weight_kg: Optional[float] = Field(None, description="Weight in kg")'),
    ('fragile: bool = Field(False, description="sample")', 'fragile: bool = Field(False, description="Whether the object is fragile")'),
    ('description="2D sample"', 'description="2D bounding box"'),
    # GraspExperience
    ('id: str = Field(..., description="sample")', 'id: str = Field(..., description="Unique experience ID")'),
    ('description="sample")\n    \n    # Note\n\n    object_info: ObjectInfo = Field(..., description="target_objectEN")',
     'description="Timestamp")\n    \n    # Note\n\n    object_info: ObjectInfo = Field(..., description="Target object information")'),
    ('description="sceneimagepath"', 'description="Scene image file path"'),
    ('description="objectimagespath"', 'description="Object crop image file path"'),
    ('description="imagespath"', 'description="Depth image file path"'),
    ('grasp_pose: GraspPose = Field(..., description="grasp_pose")', 'grasp_pose: GraspPose = Field(..., description="6-DOF grasp pose")'),
    ('description="confidence 0-1")\n    \n    # Note\n\n    gripper_type: str = Field("parallel_jaw", description="gripperclassEN")',
     'description="Confidence score (0-1)")\n    \n    # Note\n\n    gripper_type: str = Field("parallel_jaw", description="Gripper type")'),
    ('description="task_description"', 'description="Task description"'),
    ('metadata: Dict[str, Any] = Field(default_factory=dict, description="sample")', 'metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")'),
])


# =============================================================================
# 3. knowledge_base/vector_store.py
# =============================================================================
print("\n=== vector_store.py ===")
fix("knowledge_base/vector_store.py", [
    ('print(f"[ChromaVectorStore] sample {self.config.persist_directory}")',
     'print(f"[ChromaVectorStore] Initialized at {self.config.persist_directory}")'),
    ('print(f"[ChromaVectorStore] sample \'{self.config.collection_name}\' sample {self.collection.count()} sample")',
     'print(f"[ChromaVectorStore] Collection \'{self.config.collection_name}\' has {self.collection.count()} entries")'),
    ('raise ImportError("sample chromadb: pip install chromadb")',
     'raise ImportError("chromadb is required: pip install chromadb")'),
    # Two different "added/deleted" prints - need context
    ('print(f"[ChromaVectorStore] sample {len(ids)} sample")\n        \n    def query_by_text',
     'print(f"[ChromaVectorStore] Added {len(ids)} entries")\n        \n    def query_by_text'),
    ('print(f"[ChromaVectorStore] sample {len(ids)} sample")\n        \n    def count',
     'print(f"[ChromaVectorStore] Deleted {len(ids)} entries")\n        \n    def count'),
    ('print(f"[ChromaVectorStore] sample")',
     'print(f"[ChromaVectorStore] Store reset complete")'),
])


# =============================================================================
# 4. knowledge_base/grasp_memory.py
# =============================================================================
print("\n=== grasp_memory.py ===")
fix("knowledge_base/grasp_memory.py", [
    ('print(f"[GraspMemory] sample {len(all_experiences)} experiencesEN {filepath}")',
     'print(f"[GraspMemory] Exported {len(all_experiences)} experiences to {filepath}")'),
    ('print(f"[GraspMemory] sample {count} experiences")',
     'print(f"[GraspMemory] Loaded {count} experiences")'),
    ('name: str = "sample",', 'name: str = "test_object",'),
    ('description: str = "sample",', 'description: str = "A sample test object",'),
])


# =============================================================================
# 5. core/multimodal_rag.py
# =============================================================================
print("\n=== multimodal_rag.py ===")
fix("core/multimodal_rag.py", [
    ('geometry_keywords = ["sample", "sample", "sample", "sample", "width", "sample",',
     'geometry_keywords = ["shape", "cylinder", "sphere", "cube", "width", "length",'),
    ('visual_keywords = ["sample", "sample", "sample", "sample", "color", "sample",',
     'visual_keywords = ["appearance", "surface", "shiny", "matte", "color", "pattern",'),
    ('function_keywords = ["sample", "sample", "utils", "sample", "sample",',
     'function_keywords = ["grasp", "hold", "pour", "cut", "stack",'),
    ('print(f"[CrossModal] queryEN {source_modality} sample")',
     'print(f"[CrossModal] Cross-modal query: {source_modality} matching")'),
])


# =============================================================================
# 6. core/scene_graph_rag.py
# =============================================================================
print("\n=== scene_graph_rag.py ===")
fix("core/scene_graph_rag.py", [
    ('"name": "sample",\n            "category": "cup"',
     '"name": "coffee cup",\n            "category": "cup"'),
    ('sample: empty, filled, open, closed, stable, unstable',
     'Object states: empty, filled, open, closed, stable, unstable'),
    ('name="sample",\n             category="table"',
     'name="dining table",\n             category="table"'),
    ('name="sample",\n             category="cup"',
     'name="coffee cup",\n             category="cup"'),
    ('is_target=(target_object and "sample" in target_object)',
     'is_target=(target_object and "cup" in target_object.lower())'),
    ('name="sample",\n             category="book"',
     'name="notebook",\n             category="book"'),
    # constraint descriptions
    ('description=f"objectEN {target.contains}, sample"',
     'description=f"Contains {target.contains}, handle with care"'),
    ('description=f"targetEN {supported_obj.name}, grasp sample"',
     'description=f"Supports {supported_obj.name}, clear before grasping"'),
    ('description=f"targetEN {occluder.name} sample"',
     'description=f"Occluded by {occluder.name}, adjust approach"'),
    ('description=f"sample {neighbor.name} (sample {rel.distance*100:.0f}cm)"',
     'description=f"Near {neighbor.name} (distance {rel.distance*100:.0f}cm)"'),
    # Context building lines
    ('lines.append(f"### target_object\\n- sample: {target.name}\\n- classEN: {target.category}")',
     'lines.append(f"### Target Object\\n- Name: {target.name}\\n- Category: {target.category}")'),
    ('lines.append(f"- sample: {target.state.value}")',
     'lines.append(f"- State: {target.state.value}")'),
    ('lines.append(f"- sample: {target.contains}")',
     'lines.append(f"- Contains: {target.contains}")'),
    ('lines.append("\\n### ⚠️ grasp sample")',
     'lines.append("\\n### ⚠️ Grasp Constraints")'),
    ('lines.append(f"   → sample: {c.suggested_action}")',
     'lines.append(f"   → Suggestion: {c.suggested_action}")'),
])


# =============================================================================
# 7. core/unified_agentic_rag.py
# =============================================================================
print("\n=== unified_agentic_rag.py ===")
fix("core/unified_agentic_rag.py", [
    ('lines.append(f"### experience {i+1} (sample: {exp[\'source\']}, sample: {exp[\'weight\']:.2f})")',
     'lines.append(f"### Experience {i+1} (source: {exp[\'source\']}, weight: {exp[\'weight\']:.2f})")'),
    ('"grasp sample, sample",', '"Insufficient grasp force, try firmer grip",'),
    ('"sample, sample",', '"Approach path blocked, try side approach",'),
    ('"targetpositionEN, sample",', '"Target position error, refine localization",'),
    ('"objectEN, sample",', '"Object slipped, increase grip friction",'),
    ('text="grasp sample",', 'text="Grasp the target object",'),
    ('print(f"   sample: {result.attempts}")', 'print(f"   Attempts: {result.attempts}")'),
    ('print(f"\\n📝 sample:")', 'print(f"\\n📝 Summary:")'),
])


# =============================================================================
# 8. core/unified_retriever.py
# =============================================================================
print("\n=== unified_retriever.py ===")
fix("core/unified_retriever.py", [
    ('print("[Warning] HAA-RAG sample VLM sample, sample Baseline")',
     'print("[Warning] HAA-RAG requires VLM client, falling back to Baseline")'),
    ('print(f"[UnifiedRetriever] sample: {retriever_type.value}")',
     'print(f"[UnifiedRetriever] Active retriever: {retriever_type.value}")'),
])


# =============================================================================
# 9. core/agentic_grasp_pipeline.py
# =============================================================================
print("\n=== agentic_grasp_pipeline.py ===")
fix("core/agentic_grasp_pipeline.py", [
    # REFLECTION_PROMPT template
    ('- sample: {material}\n- sample: {is_fragile}',
     '- Material: {material}\n- Fragile: {is_fragile}'),
    ('## grasp sample\n- graspposition: ({px:.3f}, {py:.3f}, {pz:.3f})\n- grasp sample: {approach_direction}\n- gripper_width: {gripper_width:.3f}m\n- sample: {force_level}\n- sample: {considerations}',
     '## Grasp Plan\n- Position: ({px:.3f}, {py:.3f}, {pz:.3f})\n- Approach direction: {approach_direction}\n- Gripper width: {gripper_width:.3f}m\n- Force level: {force_level}\n- Considerations: {considerations}'),
    ('## experiencesEN\n{similar_experiences}',
     '## Similar Experiences\n{similar_experiences}'),
    ('## sample:\n1. **sample**: sample？gripperobjectsEN？\n2. **sample**: sample？sample？\n3. **sample**: experiencesEN？sample？',
     '## Review Points:\n1. **Grasp stability**: Is the gripper properly matched to the object?\n2. **Force appropriateness**: Is the force level suitable for this material?\n3. **Experience alignment**: Are similar experiences properly utilized?'),
    # _format_experiences
    ('f"sample={exp.get(\'force_level\', \'normal\')}")', 'f"force={exp.get(\'force_level\', \'normal\')}")'),
    # ANALYSIS_PROMPT
    ('## grasp sample\n- object: {object_name}\n- Position: ({px:.3f}, {py:.3f}, {pz:.3f})\n- sample: {force_level}\n- gripper_width: {gripper_width:.3f}m',
     '## Grasp Parameters\n- Object: {object_name}\n- Position: ({px:.3f}, {py:.3f}, {pz:.3f})\n- Force level: {force_level}\n- Gripper width: {gripper_width:.3f}m'),
    ('## failureEN\n{failure_description}\n\n## sample\n{sensor_feedback}',
     '## Failure Description\n{failure_description}\n\n## Sensor Feedback\n{sensor_feedback}'),
    ('"root_cause": "sample"', '"root_cause": "Detailed root cause analysis"'),
    ('failure_description=failure_info.get("description", "sample")',
     'failure_description=failure_info.get("description", "No description available")'),
    ('key_considerations=plan.key_considerations + ["sample"]',
     'key_considerations=plan.key_considerations + ["Adjusted based on failure analysis"]'),
    ('print(f"[ExperienceEvolver] sample {len(experience_ids)} experiences")',
     'print(f"[ExperienceEvolver] Updated {len(experience_ids)} experiences")'),
    # sceneEN
    ('## sceneEN', '## Scene Context'),
    ('objectdescription:', '- Description:'),
])


# =============================================================================
# 10. core/affordance_rag.py
# =============================================================================
print("\n=== affordance_rag.py ===")
fix("core/affordance_rag.py", [
    ('"secondary_affordances": ["sample"]', '"secondary_affordances": ["liftable"]'),
    ('"material": "sample"', '"material": "unknown"'),
    ('"avoid_regions": ["sample"]', '"avoid_regions": ["fragile_area"]'),
    # Affordance guide text
    ('- graspable_body: sample\n- graspable_edge: sample\n- pinchable: sample\n- clampable: sample',
     '- graspable_body: Can be grasped around the main body\n- graspable_edge: Can be grasped at the edge/rim\n- pinchable: Can be pinched with fingertips\n- clampable: Can be clamped firmly'),
    ('- fragile: sample\n- deformable: sample\n- liquid_container: sample: ceramic, glass, ...',
     '- fragile: Handle with reduced force\n- deformable: Shapes may change under pressure\n- liquid_container: Contains liquid, keep upright. Materials: ceramic, glass, ...'),
    ('raise ValueError("sample JSON")', 'raise ValueError("Invalid JSON response from VLM")'),
    # Category matching
    ('if any(w in desc_lower for w in ["sample", "cup", "mug"]):',
     'if any(w in desc_lower for w in ["cup", "mug", "goblet"]):'),
    ('elif any(w in desc_lower for w in ["sample", "bottle"]):',
     'elif any(w in desc_lower for w in ["bottle", "flask", "jar"]):'),
    ('elif any(w in desc_lower for w in ["sample", "phone"]):',
     'elif any(w in desc_lower for w in ["tablet", "phone", "laptop"]):'),
    # Print statements
    ('print(f"[HAA-RAG] sample(classEN)retrieval: {len(category_results)} sample")',
     'print(f"[HAA-RAG] Category retrieval: {len(category_results)} results")'),
    ('print(f"[HAA-RAG] sample(sample)sample: {len(affordance_results)} sample")',
     'print(f"[HAA-RAG] Affordance filtering: {len(affordance_results)} results")'),
    ('print(f"[HAA-RAG] sample(sample)sample: {len(visual_results)} sample")',
     'print(f"[HAA-RAG] Visual reranking: {len(visual_results)} results")'),
    ('print(f"[HAA-RAG] retrievaldone, ... returns {len(final_results)} sample")',
     'print(f"[HAA-RAG] Retrieval complete, returning {len(final_results)} results")'),
    ('"description": f"... sample={result.final_score:.2f}"',
     '"description": f"Score={result.final_score:.2f}"'),
    ('base_strategy += "\\n⚠️ objectEN, sample, ENgripper_width"',
     'base_strategy += "\\n⚠️ Fragile object detected, reduce force and adjust gripper width"'),
])


# =============================================================================
# 11. agent/react_engine.py
# =============================================================================
print("\n=== react_engine.py ===")
fix("agent/react_engine.py", [
    # _simulate_analysis templates
    ('"General adjustment": {"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"},\n            "General adjustment": {"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"},\n            "General adjustment": {"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"},\n            "utils": {"material": "General adjustment", "fragile": "General adjustment", "grip_area": "General adjustment"},\n            "General adjustment": {"material": "General adjustment", "fragile": "General adjustment", "grip_area": "General adjustment"},',
     '"cup": {"material": "ceramic/glass", "fragile": "true", "grip_area": "handle"},\n            "bottle": {"material": "plastic/glass", "fragile": "varies", "grip_area": "body"},\n            "tool": {"material": "metal/rubber", "fragile": "false", "grip_area": "handle"},\n            "utils": {"material": "mixed", "fragile": "false", "grip_area": "body"},\n            "default": {"material": "unknown", "fragile": "unknown", "grip_area": "center"},'),
    ('"center_of_mass": "General adjustment"', '"center_of_mass": "estimated_center"'),
    ('"material": "General adjustment",\n            "fragile": "General adjustment",\n            "grip_area": "objectEN"',
     '"material": "unknown",\n            "fragile": "unknown",\n            "grip_area": "center"'),
    # PlanGraspTool parameters
    ('"object_analysis": "objectENresult"', '"object_analysis": "Object analysis result"'),
    ('"reference_experiences": "experiences"', '"reference_experiences": "Retrieved similar experiences"'),
    ('"constraints": "sample (sample)"', '"constraints": "Task constraints (e.g. workspace limits)"'),
    # PlanGraspTool output
    ('"notes": "objectsENresultsretrieved_experiencesEN"', '"notes": "Grasp plan based on object analysis and retrieved experiences"'),
    # SimulateGraspTool parameters
    ('"grasp_plan": "graspplanning result"', '"grasp_plan": "Grasp planning result"'),
    ('"object_properties": "objectEN"', '"object_properties": "Object properties"'),
    # Warnings
    ('result["warnings"].append("General adjustment")\n        if not checks["force_appropriate"]:\n            result["warnings"].append("General adjustment")\n        if not checks["stability"]:\n            result["warnings"].append("grasp sample")',
     'result["warnings"].append("Potential collision detected")\n        if not checks["force_appropriate"]:\n            result["warnings"].append("Force may be inappropriate")\n        if not checks["stability"]:\n            result["warnings"].append("Grasp stability is low")'),
    # ReflectFailureTool parameters
    ('"failure_type": "failureclassEN (slip/collision/unreachable/width_mismatchEN)"', '"failure_type": "Failure type (slip/collision/unreachable/width_mismatch)"'),
    ('"execution_context": "executeEN"', '"execution_context": "Execution context details"'),
    ('"previous_attempts": "General adjustment"', '"previous_attempts": "Previous attempt history"'),
    # Failure analysis suggestions
    ('"sample (+20%)"', '"Increase grip force (+20%)"'),
    ('"suggestions": [\n                    "General adjustment",\n                    "Adjust grasp strategy",\n                    "Suboptimal approach path"',
     '"suggestions": [\n                    "Adjust approach direction",\n                    "Adjust grasp strategy",\n                    "Suboptimal approach path"'),
    ('analysis["suggestions"].insert(0, "Try alternative grasp approach")',
     'analysis["suggestions"].insert(0, "Try alternative grasp approach")'),
    # force_damage suggestions  
    ('"Reduce force by 30%",\n                    "General adjustment",\n                    "Excessive force applied"',
     '"Reduce force by 30%",\n                    "Use force-sensitive grip mode",\n                    "Switch to deformable object strategy"'),
    # default fallback
    ('"suggestions": ["General adjustment", "General adjustment"]', '"suggestions": ["Re-analyze the failure", "Try different grasp approach"]'),
    # REACT_PROMPT
    ('ENgraspplanningAgent, ENReActENinferenceEN。', 'You are a grasp planning Agent. Use the ReAct reasoning framework to solve the task.'),
    # Error observation  
    ('trace.add_step(thought, "error", {}, "General adjustment")', 'trace.add_step(thought, "error", {}, "No valid action found")'),
    # Simulated responses  
    ('Action Input: {"object_name": "General adjustment", "focus_aspects": ["material", "size", "fragility"]}',
     'Action Input: {"object_name": "target_object", "focus_aspects": ["material", "size", "fragility"]}'),
    ('Action Input: {"object_analysis": "sample, sample", "reference_experiences": "Retrieved grasp experiences applied"}',
     'Action Input: {"object_analysis": "Analyzed object properties", "reference_experiences": "Retrieved grasp experiences applied"}'),
    # print at bottom
    ('output.append(f"   sample: {r.metadata.get(\'notes\', \'N/A\')}")',
     'output.append(f"   Notes: {r.metadata.get(\'notes\', \'N/A\')}")'),
    ('task = "planningENgrasp sample"', 'task = "Plan an optimal grasp for the target object"'),
    # Result lines
    ('"grasppathEN"', '"Collision in grasp approach path"'),
])


# =============================================================================
# 12. agent/reflection.py
# =============================================================================
print("\n=== reflection.py ===")
fix("agent/reflection.py", [
    # FAILURE_CAUSE_MAP - SLIP
    ('"General factor",\n                "Slippery object surface",\n                "grasp sample",\n                "Slippery object surface"',
     '"Insufficient grip force",\n                "Slippery object surface",\n                "Incorrect grasp point",\n                "Surface friction mismatch"'),
    # COLLISION
    ('"pathplanningEN",\n                "General factor",\n                "General factor",\n                "General factor"',
     '"Path planning error",\n                "Obstacle proximity misjudged",\n                "Workspace boundary violation",\n                "Dynamic obstacle appeared"'),
    # WIDTH_MISMATCH
    ('"Slippery object surface",\n                "grasp sample",\n                "Slippery object surface",\n                "gripper_width"',
     '"Size estimation inaccurate",\n                "Incorrect grasp width",\n                "Object deformed during approach",\n                "Gripper width limits exceeded"'),
    # FORCE DAMAGE
    ('"General factor",\n                "Slippery object surface",\n                "General factor",\n                "grasp sample"',
     '"Excessive applied force",\n                "Material fragility misjudged",\n                "Force sensor calibration error",\n                "Grip force too high for object"'),
    # UNREACHABLE
    ('"Target position unreachable",\n                "General factor",\n                "Invalid approach path",\n                "General factor"',
     '"Target position unreachable",\n                "Joint limits exceeded",\n                "Invalid approach path",\n                "Workspace boundary exceeded"'),
    # DROP
    ('"grasp sample",\n                "General factor",\n                "General factor",\n                "Slippery object surface"',
     '"Grasp stability insufficient",\n                "Inertial forces during motion",\n                "Grip relaxation detected",\n                "Object weight underestimated"'),
    # _determine_root_cause
    ('root_cause = "General factor"\n                confidence = 0.85\n            contributing_factors = ["General factor", "grasp sample"]',
     'root_cause = "Insufficient grip force"\n                confidence = 0.85\n            contributing_factors = ["Low surface friction", "Incorrect grasp point"]'),
    ('contributing_factors = ["General factor", "General factor"]\n            \n        elif failure_type == FailureType.COLLISION:\n            if context.obstacles_nearby:\n                root_cause = f"sample({', 
     'contributing_factors = ["Width estimation error", "Sensor calibration needed"]\n            \n        elif failure_type == FailureType.COLLISION:\n            if context.obstacles_nearby:\n                root_cause = f"Collision with nearby objects ({'),
    ("})sample\"", '})\"'),
    ('contributing_factors = ["General factor", "pathplanning"]',
     'contributing_factors = ["Obstacle proximity", "Path planning error"]'),
    ('contributing_factors = ["General factor", "General factor"]\n            \n        # Note',
     'contributing_factors = ["Excessive force applied", "Material sensitivity misjudged"]\n            \n        # Note'),
    # _generate_adjustments speed
    ('"reason": "General factor",', '"reason": "Reduce collision risk",'),
    # _generate_alternatives
    ('FailureType.SLIP: [\n                "Adjust grasp strategy",\n                "General factor",\n                "General factor",',
     'FailureType.SLIP: [\n                "Adjust grasp strategy",\n                "Increase surface friction contact",\n                "Use wrap grip instead of pinch",'),
    ('FailureType.COLLISION: [\n                "General factor",\n                "General factor",\n                "Adjust grasp strategy",',
     'FailureType.COLLISION: [\n                "Adjust approach path",\n                "Clear nearby obstacles",\n                "Adjust grasp strategy",'),
    ('FailureType.WIDTH_MISMATCH: [\n                "Excessive force applied",\n                "Try different grasp type",',
     'FailureType.WIDTH_MISMATCH: [\n                "Re-estimate object dimensions",\n                "Try different grasp type",'),
    ('FailureType.FORCE_DAMAGE: [\n                "Excessive force applied",\n                "General factor",\n                "General factor",',
     'FailureType.FORCE_DAMAGE: [\n                "Reduce grasp force",\n                "Use softer grip type",\n                "Apply force sensing feedback",'),
    ('FailureType.DROP: [\n                "Adjust grasp strategy",\n                "Try alternative grasp approach",\n                "General factor",',
     'FailureType.DROP: [\n                "Adjust grasp strategy",\n                "Try alternative grasp approach",\n                "Increase grip security",'),
    ('base_alternatives = alternative_map.get(failure_type, ["General factor"])',
     'base_alternatives = alternative_map.get(failure_type, ["Try alternative approach"])'),
    ('alternatives.insert(0, "General factor")',
     'alternatives.insert(0, "Request human guidance")'),
    # _compute_adjusted_parameters
    ('if "General factor" in alt:\n                    params["grasp_type"] = "wrap"\n                    params["gripper_width"] = context.gripper_width * 1.5\n                elif "General factor" in alt:',
     'if "wrap" in alt.lower() or "friction" in alt.lower():\n                    params["grasp_type"] = "wrap"\n                    params["gripper_width"] = context.gripper_width * 1.5\n                elif "approach" in alt.lower() or "path" in alt.lower():'),
    # _extract_lesson
    ('f"{param}sample{change.get(\'old\')}sample{change.get(\'new\')}"',
     'f"{param}: {change.get(\'old\')} -> {change.get(\'new\')}"'),
    ('"argsEN: "', '"Parameter adjustments: "'),
    ('f"object\'{context.object_name}\'({context.object_category}): "\n            f"failureEN{analysis.root_cause}"',
     'f"Object \'{context.object_name}\' ({context.object_category}): "\n            f"Failure cause: {analysis.root_cause}"'),
    # VLM prompt
    ('-   Gripper width:', '- Gripper width:'),
    ('- Initial analysis: {context.grasp_force}N', '- Grasp force: {context.grasp_force}N'),
    ('Initial analysis:\n- Initial analysis: {context.force_readings}\n- Initial analysis: {context.slip_detected}\n- Initial analysis: {context.contact_detected}',
     'Sensor readings:\n- Force readings: {context.force_readings}\n- Slip detected: {context.slip_detected}\n- Contact detected: {context.contact_detected}'),
    ('Initial analysis: {context.attempt_number}\n\nInitial analysis:\n1. Root cause of the failure\n2. sample\n3. sample',
     'Attempt number: {context.attempt_number}\n\nPlease analyze:\n1. Root cause of the failure\n2. Contributing factors\n3. Recommended adjustments'),
    # if __main__ demo
    ('object_name="General factor",', 'object_name="glass_cup",'),
    # {"causes": ...} fallback
    ('{"causes": ["General factor"], "indicators": []}',
     '{"causes": ["Unknown failure cause"], "indicators": []}'),
    # Bottom prints
    ('f"\\nEN: {result.analysis.contributing_factors}"',
     'f"\\nContributing factors: {result.analysis.contributing_factors}"'),
    ('f"\\nEN:"', 'f"\\nAdjustments:"'),
    ('f"\\nEN: {result.analysis.alternative_strategies[:3]}"',
     'f"\\nAlternative strategies: {result.analysis.alternative_strategies[:3]}"'),
    ('f"\\nEN: {result.should_retry}"', 'f"\\nShould retry: {result.should_retry}"'),
    ('f"Initial analysis: {result.retry_strategy}"', 'f"Retry strategy: {result.retry_strategy}"'),
    ('f"successEN: {result.success_probability:.2f}"', 'f"Success probability: {result.success_probability:.2f}"'),
    ('f"\\nEN: {result.lesson_learned}"', 'f"\\nLesson learned: {result.lesson_learned}"'),
    ('f"Initial analysis: {result.analysis.root_cause}"', 'f"Root cause: {result.analysis.root_cause}"'),
])


# =============================================================================
# 13. agent/retry_strategy.py
# =============================================================================
print("\n=== retry_strategy.py ===")
fix("agent/retry_strategy.py", [
    ('return False, f"sample({self._config.max_attempts})"',
     'return False, f"Maximum attempts reached ({self._config.max_attempts})"'),
    ('return False, f"failureclassEN({context.failure_type.value})sample"',
     'return False, f"Failure type ({context.failure_type.value}) is not retryable"'),
    ('return True, "sample"', 'return True, "Retry conditions met"'),
    ('print(f"\\n[Retry] sample {attempt} sample...")', 'print(f"\\n[Retry] Attempt {attempt} starting...")'),
    ('print(f"[Retry] sample: {plan.get(\'reasoning\', \'\')}")', 'print(f"[Retry] Reasoning: {plan.get(\'reasoning\', \'\')}")'),
    ('print(f"[Retry] sample: {plan[\'strategy_name\']} (Level: {plan[\'retry_level\']})")',
     'print(f"[Retry] Strategy: {plan[\'strategy_name\']} (Level: {plan[\'retry_level\']})")'),
    ('lessons.append(f"success: {plan[\'strategy_name\']}sample")',
     'lessons.append(f"Success with strategy: {plan[\'strategy_name\']}")'),
    ('object_name="sample",', 'object_name="test_cup",'),
    ('print(f"sample: {result.total_attempts}")', 'print(f"Total attempts: {result.total_attempts}")'),
])


# =============================================================================
# 14. agent/experience_learner.py
# =============================================================================
print("\n=== experience_learner.py ===")
fix("agent/experience_learner.py", [
    ('f"sample: {json.dumps(param_changes, ensure_ascii=False)}",',
     'f"Parameter changes: {json.dumps(param_changes, ensure_ascii=False)}",'),
    ('f"sample: {original_failure.failure_cause}",',
     'f"Original failure: {original_failure.failure_cause}",'),
    ('caution = f"sample: {failure_cause}, sample: {json.dumps(adjustments, ensure_ascii=False)}"',
     'caution = f"Failure cause: {failure_cause}, Adjustments: {json.dumps(adjustments, ensure_ascii=False)}"'),
    ('f"width{experience.gripper_width:.3f}m, sample{experience.grasp_force:.1f}N"',
     'f"Width: {experience.gripper_width:.3f}m, Force: {experience.grasp_force:.1f}N"'),
    ('f"sample: {experience.failure_cause}"',
     'f"Failure cause: {experience.failure_cause}"'),
    ('lessons.append(f"sample: {adj_str}")',
     'lessons.append(f"Adjustments: {adj_str}")'),
    ('"width_mismatch": "grasp sampleobjectEN",',
     '"width_mismatch": "Re-estimate object width for grasp",'),
    # Demo section
    ('object_name="sample",\n            object_category="cup",',
     'object_name="ceramic_cup",\n            object_category="cup",'),
    ('print(f"   sample: +{result1.weight_change}")', 'print(f"   Weight change: +{result1.weight_change}")'),
    ('print(f"   sample: {result1.lessons_extracted}")', 'print(f"   Lessons: {result1.lessons_extracted}")'),
    ('object_name="sample",\n            object_category="tool",',
     'object_name="screwdriver",\n            object_category="tool",'),
    ('print(f"   sample: {result2.weight_change}")', 'print(f"   Weight change: {result2.weight_change}")'),
    ('print(f"   sample: {result2.lessons_extracted}")', 'print(f"   Lessons: {result2.lessons_extracted}")'),
    ('object_name="sample",\n            object_category="food",',
     'object_name="banana",\n            object_category="food",'),
    ('print(f"   sample: +{result3.weight_change}")', 'print(f"   Weight change: +{result3.weight_change}")'),
    ('print(f"   sample: {result3.lessons_extracted}")', 'print(f"   Lessons: {result3.lessons_extracted}")'),
    ('print("\\n4. sample:")', 'print("\\n4. Statistics:")'),
    ('print(f"   success: ..., sample: {stats[\'corrected_count\']}")',
     'print(f"   Corrected count: {stats[\'corrected_count\']}")'),
    ('print(f"   sample: {stats[\'average_weight\']:.2f}")',
     'print(f"   Average weight: {stats[\'average_weight\']:.2f}")'),
    ('print(f"   {category}: {len(rules)} sample")',
     'print(f"   {category}: {len(rules)} rules")'),
])


# =============================================================================
# 15. agent/grasp_agent.py
# =============================================================================
print("\n=== grasp_agent.py ===")
fix("agent/grasp_agent.py", [
    ('print("[GraspAgent] load VLM sample...")', 'print("[GraspAgent] Loading VLM model...")'),
    ('print(f"[GraspAgent] sample: {len(full_prompt)} sample")',
     'print(f"[GraspAgent] Prompt length: {len(full_prompt)} chars")'),
    ('print(f"[GraspAgent] VLM sample: {len(response)} sample")',
     'print(f"[GraspAgent] VLM response: {len(response)} chars")'),
])


# =============================================================================
# 16. agent/agentic_grasp.py
# =============================================================================
print("\n=== agentic_grasp.py ===")
fix("agent/agentic_grasp.py", [
    ('print(f"[Agent] sample: {state.value}")', 'print(f"[Agent] State: {state.value}")'),
    ('react_task = f"planninggrasp{object_name}({object_category})sample。task: {task}"',
     'react_task = f"Plan a grasp for {object_name} ({object_category}). Task: {task}"'),
    ('print(f"[Agent] sample ({reflect_time:.0f}ms):")', 'print(f"[Agent] Reflection ({reflect_time:.0f}ms):")'),
    ('print(f"        sample: {reflection_result.analysis.root_cause}")',
     'print(f"        Root cause: {reflection_result.analysis.root_cause}")'),
    ('print(f"        sample: {reflection_result.should_retry}")',
     'print(f"        Should retry: {reflection_result.should_retry}")'),
    ('print("[Agent] sample")', 'print("[Agent] Maximum retries reached")'),
    ('print(f"[Agent] sample: {len(attempts)}")', 'print(f"[Agent] Total attempts: {len(attempts)}")'),
    # Demo data
    ('"object_name": "sample",\n            "category": "cup"',
     '"object_name": "ceramic_mug",\n            "category": "cup"'),
    ('"object_name": "sample",\n            "category": "tool"',
     '"object_name": "screwdriver",\n            "category": "tool"'),
    ('print(f"  - sample: {result.total_attempts}")', 'print(f"  - Total attempts: {result.total_attempts}")'),
    ('print(f"  - sample: {result.lessons_learned[:2]}")', 'print(f"  - Lessons: {result.lessons_learned[:2]}")'),
    ('print("                          sample")', 'print("                          Agent Demo Complete")'),
    ('print(f"sample: {total_attempts/len(results):.1f}")',
     'print(f"Average attempts: {total_attempts/len(results):.1f}")'),
])


# =============================================================================
# 17. scripts/benchmark.py
# =============================================================================
print("\n=== benchmark.py ===")
fix("scripts/benchmark.py", [
    # First warmup
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")\n        for _ in range(warmup_runs):\n            store.query_by_text',
     'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")\n        for _ in range(warmup_runs):\n            store.query_by_text'),
    ('print(f"[INFO] sample {n_runs} benchmark...")',
     'print(f"[INFO] Running {n_runs} benchmark iterations...")'),
    ('print(f"\\n[INFO] sample ({n_images} images)...")',
     'print(f"\\n[INFO] Benchmarking ({n_images} images)...")'),
    ('print(f"[INFO] sample ({n_texts} sample)...")',
     'print(f"[INFO] Benchmarking ({n_texts} text queries)...")'),
    ('print(f"  sample: {results[\'visual_encoding\'][\'per_image_ms\']:.2f} ms/image")',
     'print(f"  Visual encoding: {results[\'visual_encoding\'][\'per_image_ms\']:.2f} ms/image")'),
    ('print(f"  sample: {results[\'text_encoding\'][\'per_text_ms\']:.2f} ms/text")',
     'print(f"  Text encoding: {results[\'text_encoding\'][\'per_text_ms\']:.2f} ms/text")'),
    ('print(f"  sample: {results[\'visual_encoding\'][\'embedding_dim\']}")',
     'print(f"  Visual dim: {results[\'visual_encoding\'][\'embedding_dim\']}")'),
    ('print(f"  sample: {results[\'text_encoding\'][\'embedding_dim\']}")',
     'print(f"  Text dim: {results[\'text_encoding\'][\'embedding_dim\']}")'),
    ('print(f"\\n[INFO] benchmark ({db_size} sample)...")',
     'print(f"\\n[INFO] Benchmarking ({db_size} entries)...")'),
    ('texts = [f"objectdescription {i}: sample {i} benchmarkobject" for i in',
     'texts = [f"Object description {i}: benchmark object {i}" for i in'),
    ('print(f"  sample {db_size} sample")',
     'print(f"  Database size: {db_size} entries")'),
    ('print(f"  sample: {db_size} sample")',
     'print(f"  Database size: {db_size} entries")'),
    ('print("benchmark (sample RAG sample)")',
     'print("End-to-end benchmark (RAG pipeline)")'),
    ('"grasp sampleobject"', '"Plan a grasp for this object"'),
    # Second warmup
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")\n        for _ in range(warmup_runs):\n            pipeline',
     'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")\n        for _ in range(warmup_runs):\n            pipeline'),
    ('print(f"[INFO] sample {n_runs} benchmark...")',
     'print(f"[INFO] Running {n_runs} benchmark iterations...")'),
])


# =============================================================================
# 18. scripts/demo_grasp.py
# =============================================================================
print("\n=== demo_grasp.py ===")
fix("scripts/demo_grasp.py", [
    ('target_object: str = "sample",', 'target_object: str = "cup",'),
    ('print(f"  Short-term memory: {stats.get(\'short_term_count\', 0)} sample")',
     'print(f"  Short-term memory: {stats.get(\'short_term_count\', 0)} entries")'),
    ('print(f"  Long-term memory: {stats.get(\'long_term_count\', 0)} sample")',
     'print(f"  Long-term memory: {stats.get(\'long_term_count\', 0)} entries")'),
    ('("sample", "grasp sample"),',
     '("cup", "Grasp the cup carefully"),'),
    ('("sample", "sample"),',
     '("screwdriver", "Pick up the screwdriver"),'),
    ('("sample", "Plan a grasp for this object"),',
     '("bottle", "Plan a grasp for this bottle"),'),
    ('parser.add_argument("--target", type=str, default="sample", help="target_object")',
     'parser.add_argument("--target", type=str, default="cup", help="Target object name")'),
])


# =============================================================================
# 19. core/vlm_engine.py
# =============================================================================
print("\n=== vlm_engine.py ===")
fix("core/vlm_engine.py", [
    ('print(f"[VLMEngine] sample: {self.config.quantization}")',
     'print(f"[VLMEngine] Quantization: {self.config.quantization}")'),
    ('print(f"[VLMEngine] sample {num_tokens} tokens, sample: {speed:.1f} t/s")',
     'print(f"[VLMEngine] Generated {num_tokens} tokens, speed: {speed:.1f} t/s")'),
])


print(f"\n{'='*60}")
print(f"TOTAL CHANGES: {total_changes}")
print(f"{'='*60}")
