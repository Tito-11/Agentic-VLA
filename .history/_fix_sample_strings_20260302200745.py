"""Third pass: fix all remaining 'sample' and mixed placeholders."""
import os
import re
import glob

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"


def fix_file(rel_path, replacements):
    filepath = os.path.join(ROOT, rel_path.replace("/", os.sep))
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new, 1)
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


fixes = {}

# ---- scripts/build_knowledge_base.py ----
# Sample dataset: replace "sample" object names/descriptions with realistic English
fixes["scripts/build_knowledge_base.py"] = [
    # Cup samples
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.35, "y": 0.10, "z": 0.12}',
     '"object_name": "red ceramic mug",\n            "category": "cup",\n            "description": "Standard ceramic mug with handle",\n            "grasp_pose": {\n                "position": {"x": 0.35, "y": 0.10, "z": 0.12}'),
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample, sample",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.08}',
     '"object_name": "large coffee cup",\n            "category": "cup",\n            "description": "Large white coffee cup, wide mouth",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.08}'),
    ('"object_name": "sample",\n            "category": "cup",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.10}',
     '"object_name": "glass water cup",\n            "category": "cup",\n            "description": "Transparent glass cup, fragile",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.10}'),
    ('"key_insight": "sample, sample",\n            "color": "transparent"',
     '"key_insight": "Fragile material, use gentle grip",\n            "color": "transparent"'),
    # Tool samples
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.15, "z": 0.05}',
     '"object_name": "screwdriver",\n            "category": "tool",\n            "description": "Phillips head screwdriver with rubber handle",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.15, "z": 0.05}'),
    ('"key_insight": "grasp sample",\n            "color": "red"',
     '"key_insight": "Grasp the handle for stable control",\n            "color": "red"'),
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.04}',
     '"object_name": "wrench",\n            "category": "tool",\n            "description": "Adjustable wrench, 8 inch",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.04}'),
    ('"key_insight": "grasp sample",\n            "color": "blue"',
     '"key_insight": "Grasp around the handle, avoid the jaw",\n            "color": "blue"'),
    ('"object_name": "sample",\n            "category": "tool",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": 0.05, "z": 0.03}',
     '"object_name": "pliers",\n            "category": "tool",\n            "description": "Needle-nose pliers with insulated handle",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": 0.05, "z": 0.03}'),
    ('"key_insight": "grasp sample",\n            "color": "silver"',
     '"key_insight": "Grasp the handle grips firmly",\n            "color": "silver"'),
    # Container samples
    ('"object_name": "sample",\n            "category": "container",\n            "description": "500ml sample"',
     '"object_name": "water bottle",\n            "category": "container",\n            "description": "500ml plastic water bottle"'),
    ('"object_name": "sample",\n            "category": "container",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.12, "z": 0.06}',
     '"object_name": "metal thermos",\n            "category": "container",\n            "description": "Stainless steel thermos with lid",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.12, "z": 0.06}'),
    ('"key_insight": "sample, sample",\n            "color": "silver"',
     '"key_insight": "Heavy container, use firm wrap grip",\n            "color": "silver"'),
    # Food samples
    ('"object_name": "sample",\n            "category": "food",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.04}',
     '"object_name": "apple",\n            "category": "food",\n            "description": "Red apple, round and smooth",\n            "grasp_pose": {\n                "position": {"x": 0.32, "y": 0.00, "z": 0.04}'),
    ('"object_name": "sample",\n            "category": "food",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.03}',
     '"object_name": "banana",\n            "category": "food",\n            "description": "Yellow banana, curved shape",\n            "grasp_pose": {\n                "position": {"x": 0.30, "y": -0.05, "z": 0.03}'),
    # Electronics samples
    ('"object_name": "sample",\n            "category": "electronics",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.08, "z": 0.01}',
     '"object_name": "smartphone",\n            "category": "electronics",\n            "description": "Flat smartphone, glass screen",\n            "grasp_pose": {\n                "position": {"x": 0.28, "y": 0.08, "z": 0.01}'),
    ('"object_name": "sample",\n            "category": "electronics",\n            "description": "sample",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.02}',
     '"object_name": "wireless mouse",\n            "category": "electronics",\n            "description": "Ergonomic wireless mouse",\n            "grasp_pose": {\n                "position": {"x": 0.25, "y": -0.10, "z": 0.02}'),
    # Print statements
    ('print(f"\\n[INFO] sample {len(samples)} experiences")',
     'print(f"\\n[INFO] Processing {len(samples)} experiences")'),
    ('print(f"\\n[{i+1}/{len(samples)}] sample: {sample[\'object_name\']}")',
     'print(f"\\n[{i+1}/{len(samples)}] Adding: {sample[\'object_name\']}")'),
    ('print(f"[INFO] sample {json_path} sample {len(data)} experiences")',
     'print(f"[INFO] Exported to {json_path}, {len(data)} experiences")'),
    ('"--collection", type=str, default="grasp_experiences", help="sample"',
     '"--collection", type=str, default="grasp_experiences", help="Collection name in vector store"'),
]

# ---- knowledge_base/schema.py ----
fixes["knowledge_base/schema.py"] = [
    ('description="X sample"', 'description="X coordinate (meters)"'),
    ('description="Y sample"', 'description="Y coordinate (meters)"'),
    ('description="Z sample"', 'description="Z coordinate (meters)"'),
    ('description="X sample"', 'description="X component"'),
    ('description="Y sample"', 'description="Y component"'),
    ('description="Z sample"', 'description="Z component"'),
    ('description="W sample"', 'description="W component (scalar)"'),
    ('description="grasp sampleposition"', 'description="Grasp target position"'),
    ('description="grasp_pose"', 'description="Grasp orientation (quaternion)"'),
    ('description="gripper_width"', 'description="Gripper opening width (meters)"'),
    ('description="objectEN"', 'description="Object name"'),
    ('description="objectclassEN', 'description="Object category (e.g.'),
    ('description="objectdescription"', 'description="Object description"'),
    ('description="sample（sample glass, metal, plastic）"', 'description="Material type (e.g. glass, metal, plastic)"'),
    ('description="sample")\n    weight_kg', 'description="Weight in kg")\n    weight_kg'),
    ('fragile: bool = Field(False, description="sample")', 'fragile: bool = Field(False, description="Whether the object is fragile")'),
    ('description="2D sample"', 'description="2D bounding box"'),
    ('id: str = Field(..., description="sample")', 'id: str = Field(..., description="Unique experience ID")'),
    ('description="sample")\n    \n    # Note\n\n    object_info', 'description="Timestamp of the experience")\n    \n    # Note\n\n    object_info'),
    ('description="target_objectEN"', 'description="Target object information"'),
    ('description="sceneimagepath"', 'description="Scene image file path"'),
    ('description="objectimagespath"', 'description="Object crop image file path"'),
    ('description="imagespath"', 'description="Depth image file path"'),
    ('description="grasp_pose")\n    \n    # Note\n\n    success', 'description="6D grasp pose")\n    \n    # Note\n\n    success'),
    ('description="confidence 0-1")\n    \n    # Note\n\n    gripper_type', 'description="Confidence score (0-1)")\n    \n    # Note\n\n    gripper_type'),
    ('description="gripperclassEN"', 'description="Gripper type"'),
    ('description="task_description"', 'description="Task description"'),
    ('metadata: Dict[str, Any] = Field(default_factory=dict, description="sample")', 'metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")'),
    ('description="confidence 0-1")\n    reasoning', 'description="Prediction confidence (0-1)")\n    reasoning'),
    ('description="inferenceEN"', 'description="Reasoning explanation"'),
    ('description="experiencesIDEN"', 'description="Referenced experience IDs"'),
    ('description="inference_time"', 'description="Inference time in ms"'),
    ('description="retrieval_time"', 'description="Retrieval time in ms"'),
]

# ---- knowledge_base/vector_store.py ----
fixes["knowledge_base/vector_store.py"] = [
    ('print(f"[ChromaVectorStore] sample {self.config.persist_directory}")',
     'print(f"[ChromaVectorStore] Initialized at {self.config.persist_directory}")'),
    ('print(f"[ChromaVectorStore] sample \'{self.config.collection_name}\' sample {self.collection.count()} sample")',
     'print(f"[ChromaVectorStore] Collection \'{self.config.collection_name}\' has {self.collection.count()} entries")'),
    ('raise ImportError("sample chromadb: pip install chromadb")',
     'raise ImportError("chromadb is required: pip install chromadb")'),
    ('print(f"[ChromaVectorStore] sample {len(ids)} sample")\n        \n    def query_by_text',
     'print(f"[ChromaVectorStore] Added {len(ids)} entries")\n        \n    def query_by_text'),
    ('print(f"[ChromaVectorStore] sample {len(ids)} sample")\n        \n    def count',
     'print(f"[ChromaVectorStore] Deleted {len(ids)} entries")\n        \n    def count'),
]

# ---- knowledge_base/grasp_memory.py ----
fixes["knowledge_base/grasp_memory.py"] = [
    ('name: str = "sample",', 'name: str = "test_object",'),
    ('description: str = "sample",', 'description: str = "A sample object for testing",'),
]

# ---- core/multimodal_rag.py ----
fixes["core/multimodal_rag.py"] = [
    ('geometry_keywords = ["sample", "sample", "sample", "sample", "width", "sample",',
     'geometry_keywords = ["shape", "cylinder", "sphere", "cube", "width", "length",'),
    ('visual_keywords = ["sample", "sample", "sample", "sample", "color", "sample",',
     'visual_keywords = ["appearance", "surface", "shiny", "matte", "color", "pattern",'),
    ('function_keywords = ["sample", "sample", "utils", "sample", "sample",',
     'function_keywords = ["grasp", "hold", "pour", "cut", "stack",'),
]

# ---- core/scene_graph_rag.py ----
fixes["core/scene_graph_rag.py"] = [
    ('"objectsEN', '"Analyze the objects in the scene and their spatial relationships'),
    ('sample: empty, filled, open, closed, stable, unstable',
     'Object states: empty, filled, open, closed, stable, unstable'),
    ('"name": "sample",\n            "category": "cup"',
     '"name": "coffee cup",\n            "category": "cup"'),
    ('name="sample",\n             category="table"',
     'name="dining table",\n             category="table"'),
    ('name="sample",\n             category="cup"',
     'name="coffee cup",\n             category="cup"'),
    ('is_target=(target_object and "sample" in target_object)',
     'is_target=(target_object and "cup" in target_object.lower())'),
    ('name="sample",\n             category="book"',
     'name="notebook",\n             category="book"'),
    ('description=f"sample {neighbor.name} (sample {rel.distance*100:.0f}cm)"',
     'description=f"Near {neighbor.name} (distance {rel.distance*100:.0f}cm)"'),
]

# ---- core/unified_agentic_rag.py ----
fixes["core/unified_agentic_rag.py"] = [
    ('"retrievalexperiences"', '"Retrieved Experiences"'),
    ('f"### experience {i+1} (sample: {exp[\'source\']}, sample: {exp[\'weight\']:.2f})"',
     'f"### Experience {i+1} (source: {exp[\'source\']}, weight: {exp[\'weight\']:.2f})"'),
    ('"graspclassEN"', '"Grasp type"'),
    ('"sceneEN"', '"Scene Constraints"'),
    ('"grasp sample, sample"', '"Insufficient grasp force, try firmer grip"'),
    ('"sample, sample"', '"Approach path blocked, try side approach"'),
    ('"targetpositionEN, sample"', '"Target position error, refine localization"'),
    ('"objectEN, sample"', '"Object slipped, increase friction grip"'),
]

# ---- core/unified_retriever.py ----
fixes["core/unified_retriever.py"] = [
    ('print("[Warning] HAA-RAG sample VLM sample, sample Baseline")',
     'print("[Warning] HAA-RAG requires VLM client, falling back to Baseline")'),
]

# ---- core/agentic_grasp_pipeline.py ----
fixes["core/agentic_grasp_pipeline.py"] = [
    ('"root_cause": "sample"', '"root_cause": "Unknown failure cause"'),
    ('failure_info.get("description", "sample")', 'failure_info.get("description", "No description available")'),
    ('key_considerations=plan.key_considerations + ["sample"]',
     'key_considerations=plan.key_considerations + ["Consider alternative approach"]'),
    ('print(f"[Reflection] sample, sample: {reflection_result.issues}")',
     'print(f"[Reflection] Issues found: {reflection_result.issues}")'),
    ('key_considerations=["sample"]', 'key_considerations=["Use default grasp strategy"]'),
    # Fix the suggestion parsing section
    ('if "sample" in suggestion or "force" in suggestion_lower:',
     'if "force" in suggestion_lower or "strength" in suggestion_lower:'),
    ('if "sample" in suggestion or "sample" in suggestion:\n                    adjustments["force_adjustment"] = "increase"',
     'if "increase" in suggestion_lower or "firm" in suggestion_lower:\n                    adjustments["force_adjustment"] = "increase"'),
    ('elif "sample" in suggestion or "sample" in suggestion:\n                    adjustments["force_adjustment"] = "decrease"',
     'elif "decrease" in suggestion_lower or "gentle" in suggestion_lower:\n                    adjustments["force_adjustment"] = "decrease"'),
    ('if "sample" in suggestion or "side" in suggestion_lower:',
     'if "approach" in suggestion_lower or "side" in suggestion_lower:'),
    ('if "sample" in suggestion or "sample" in suggestion:\n                    adjustments["gripper_adjustment"] = 0.01',
     'if "wider" in suggestion_lower or "open" in suggestion_lower:\n                    adjustments["gripper_adjustment"] = 0.01'),
    ('elif "sample" in suggestion or "sample" in suggestion:\n                    adjustments["gripper_adjustment"] = -0.01',
     'elif "narrow" in suggestion_lower or "close" in suggestion_lower:\n                    adjustments["gripper_adjustment"] = -0.01'),
]

# ---- core/affordance_rag.py ----
fixes["core/affordance_rag.py"] = [
    ('["sample", "cup", "mug"]', '["cup", "mug", "goblet"]'),
    ('["sample", "bottle"]', '["bottle", "flask"]'),
    ('["sample", "phone"]', '["tablet", "phone"]'),
    ('print(f"[HAA-RAG] sample(classEN)retrieval: {len(category_results)} sample")',
     'print(f"[HAA-RAG] Category retrieval: {len(category_results)} results")'),
    ('print(f"[HAA-RAG] sample(sample)sample: {len(affordance_results)} sample")',
     'print(f"[HAA-RAG] Affordance filtering: {len(affordance_results)} results")'),
    ('print(f"[HAA-RAG] sample(sample)sample: {len(visual_results)} sample")',
     'print(f"[HAA-RAG] Visual reranking: {len(visual_results)} results")'),
    ('"secondary_affordances": ["sample"]', '"secondary_affordances": ["liftable"]'),
    ('"material": "sample"', '"material": "unknown"'),
    ('"avoid_regions": ["sample"]', '"avoid_regions": ["fragile_area"]'),
]

# ---- agent/agentic_grasp.py ----
fixes["agent/agentic_grasp.py"] = [
    ('"object_name": "sample",', '"object_name": "test_object",'),
    ('"object_name": "sample",', '"object_name": "test_object_2",'),
]

# ---- agent/experience_learner.py ----
fixes["agent/experience_learner.py"] = [
    ('caution = f"sample: {failure_cause}, sample: {json.dumps(adjustments, ensure_ascii=False)}"',
     'caution = f"Failure cause: {failure_cause}, adjustments: {json.dumps(adjustments, ensure_ascii=False)}"'),
    ('object_name="sample",\n', 'object_name="test_cup",\n'),
    ('object_name="sample",\n', 'object_name="test_mug",\n'),
    ('object_name="sample",\n', 'object_name="test_bottle",\n'),
]

# ---- agent/grasp_agent.py ----
fixes["agent/grasp_agent.py"] = [
    ('print(f"[GraspAgent] sample: {len(full_prompt)} sample")',
     'print(f"[GraspAgent] Prompt length: {len(full_prompt)} chars")'),
    ('print(f"[GraspAgent] VLM sample: {len(response)} sample")',
     'print(f"[GraspAgent] VLM response: {len(response)} chars")'),
]

# ---- agent/react_engine.py ----
fixes["agent/react_engine.py"] = [
    ('"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"',
     '"material": "ceramic/glass", "fragile": "true", "grip_area": "handle"'),
    ('"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"',
     '"material": "metal/plastic", "fragile": "false", "grip_area": "body"'),
    ('"material": "sample/sample", "fragile": "General adjustment", "grip_area": "General adjustment"',
     '"material": "mixed/other", "fragile": "varies", "grip_area": "optimal_region"'),
    ('"constraints": "sample (sample)"', '"constraints": "workspace limits (reachability)"'),
    ('"object_analysis": "sample, sample"', '"object_analysis": "round object, smooth surface"'),
    ('"grasp_plan": "Optimized grasp", "object_properties": "Analyzed"',
     '"grasp_plan": "Optimized grasp pose", "object_properties": "Properties analyzed"'),
    ('f"{param}sample{change.get(\'old\')}sample{change.get(\'new\')}"',
     'f"{param}: {change.get(\'old\')} -> {change.get(\'new\')}"'),
]

# ---- agent/reflection.py ----
fixes["agent/reflection.py"] = [
    # Remaining "sample" and "General factor" placeholders  
    ('"grasp General factor"', '"Insufficient grasp contact area"'),
    ('contributing_factors = ["General factor", "grasp General factor"]',
     'contributing_factors = ["Surface friction too low", "Insufficient grip force"]'),
    ('contributing_factors = ["General factor", "General factor"]',
     'contributing_factors = ["Width estimation error", "Sensor calibration needed"]'),
    ('root_cause = f"General factor({', 'root_cause = f"Collision with nearby objects ({'),
    ('contributing_factors = ["General factor", "pathplanning"]',
     'contributing_factors = ["Obstacle proximity", "Path planning error"]'),
    ('contributing_factors = ["General factor", "General factor"]',
     'contributing_factors = ["Excessive applied force", "Material sensitivity misjudged"]'),
    ('f"General factor{context.attempt_number}General factor"',
     'f"Repeated failure on attempt {context.attempt_number}"'),
    ('"General factor")\n        \n        # Note\n\n        if context.attempt_number >= 2',
     '"General approach adjustment")\n        \n        # Note\n\n        if context.attempt_number >= 2'),
    ('"General factor")\n            alternatives.insert(0, "Try a wider grasp approach")',
     '"Try different object region")\n            alternatives.insert(0, "Try a wider grasp approach")'),
    ('alternatives.extend(base_alternatives)\n        \n        # Note\n\n        if context.attempt_number >= 2:\n            alternatives.insert(0, "Try a wider grasp approach")\n        if context.attempt_number >= 3:\n            alternatives.insert(0, "General factor")',
     'alternatives.extend(base_alternatives)\n        \n        # Note\n\n        if context.attempt_number >= 2:\n            alternatives.insert(0, "Try a wider grasp approach")\n        if context.attempt_number >= 3:\n            alternatives.insert(0, "Request human guidance")'),
    ('"reason": "General factor"', '"reason": "Reduce impact velocity"'),
    # Fix VLM analysis prompt remaining issues
    ('graspargs:', 'Grasp parameters:'),
    ('gripperwidth:', '- Gripper width:'),
    ('- sample: {context.grasp_force}N', '- Grasp force: {context.grasp_force}N'),
    ('sample:\n- sample: {context.force_readings}', 'Sensor readings:\n- Force: {context.force_readings}'),
    ('- sample: {context.slip_detected}', '- Slip detected: {context.slip_detected}'),
    ('- sample: {context.contact_detected}', '- Contact detected: {context.contact_detected}'),
    ('sample: {context.attempt_number}', 'Attempt number: {context.attempt_number}'),
    ('sample:\n1. Root cause of the failure\n2. sample\n3. sample',
     'Please analyze:\n1. Root cause of the failure\n2. Contributing factors\n3. Recommended adjustments'),
    # Alternative strategies
    ('FailureType.SLIP: [\n                "Adjust grasp strategy",\n                "General factor",\n                "General factor",',
     'FailureType.SLIP: [\n                "Adjust grasp strategy",\n                "Increase surface friction contact",\n                "Use wrap grip instead of pinch",'),
    ('FailureType.COLLISION: [\n                "General factor",\n                "General factor",\n                "Adjust grasp strategy",',
     'FailureType.COLLISION: [\n                "Adjust approach path",\n                "Clear nearby obstacles",\n                "Adjust grasp strategy",'),
    ('FailureType.WIDTH_MISMATCH: [\n                "Excessive force applied",\n                "Try different grasp type",\n                "Adjust grasp strategy",',
     'FailureType.WIDTH_MISMATCH: [\n                "Re-estimate object dimensions",\n                "Try different grasp type",\n                "Adjust grasp strategy",'),
    ('FailureType.FORCE_DAMAGE: [\n                "Excessive force applied",\n                "General factor",\n                "General factor",',
     'FailureType.FORCE_DAMAGE: [\n                "Reduce grasp force",\n                "Use softer grip type",\n                "Apply force sensing feedback",'),
    ('FailureType.DROP: [\n                "Adjust grasp strategy",\n                "Try alternative grasp approach",\n                "General factor",',
     'FailureType.DROP: [\n                "Adjust grasp strategy",\n                "Try alternative grasp approach",\n                "Increase grip security",'),
]

# ---- agent/retry_strategy.py ----
fixes["agent/retry_strategy.py"] = [
    ('return True, "sample"', 'return True, "Retry conditions met"'),
    ('print(f"\\n[Retry] sample {attempt} sample...")', 'print(f"\\n[Retry] Attempt {attempt} starting...")'),
    ('object_name="sample"', 'object_name="test_object"'),
]

# ---- scripts/benchmark.py ----
fixes["scripts/benchmark.py"] = [
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")\n', 'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")\n'),
    ('print(f"[INFO] sample ({n_texts} sample)...")', 'print(f"[INFO] Benchmarking ({n_texts} queries)...")'),
    ('print(f"  sample {db_size} sample")', 'print(f"  Database size: {db_size} entries")'),
    ('print(f"  sample: {db_size} sample")', 'print(f"  Database size: {db_size} entries")'),
    ('print("benchmark (sample RAG sample)")', 'print("End-to-end benchmark (RAG + VLM pipeline)")'),
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")', 'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")'),
    ('print("\\n| sample | sample |")', 'print("\\n| Metric | Value |")'),
]

# ---- scripts/demo_grasp.py ----
fixes["scripts/demo_grasp.py"] = [
    ('target_object: str = "sample"', 'target_object: str = "cup"'),
    ('"Plan a grasp for this object"', '"Plan a grasp for this object"'),
]

# ---- core/vlm_engine.py ----
fixes["core/vlm_engine.py"] = [
    ('print(f"[VLMEngine] sample {num_tokens} tokens, sample: {speed:.1f} t/s")',
     'print(f"[VLMEngine] Generated {num_tokens} tokens, speed: {speed:.1f} t/s")'),
]

# Apply all fixes
total = 0
for rel_path, pairs in fixes.items():
    if fix_file(rel_path, pairs):
        print(f"Fixed: {rel_path}")
        total += 1
    else:
        print(f"WARN: No changes in {rel_path}")

print(f"\nTotal files fixed: {total}")
