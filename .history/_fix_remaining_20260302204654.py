"""Second pass: fix remaining ~54 placeholder strings missed by first script."""
import os

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"
total = 0

def fix(rel_path, pairs):
    global total
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
            print(f"  MISS: {old[:90]!r}")
    if content != orig:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        total += count
        print(f"  OK {rel_path}: {count}")

# --- scene_graph_rag.py: 3 name="sample" + 1 sceneEN ---
print("scene_graph_rag.py")
fix("core/scene_graph_rag.py", [
    ('name="sample",\n            category="table"', 'name="dining table",\n            category="table"'),
    ('name="sample",\n            category="cup"', 'name="coffee cup",\n            category="cup"'),
    ('name="sample",\n            category="book"', 'name="notebook",\n            category="book"'),
    ('lines = ["## sceneEN\\n"]', 'lines = ["## Scene Context\\n"]'),
])

# --- unified_agentic_rag.py: graspclassEN + sceneEN ---
print("unified_agentic_rag.py")
fix("core/unified_agentic_rag.py", [
    ('"graspclassEN"', '"Grasp type"'),
    ('"\\n## ⚠️ sceneEN"', '"\\n## ⚠️ Scene Constraints"'),
])

# --- agentic_grasp_pipeline.py: many remaining ---
print("agentic_grasp_pipeline.py")
fix("core/agentic_grasp_pipeline.py", [
    ('print(f"[Reflection] sample, sample: {reflection_result.issues}")',
     'print(f"[Reflection] Issues found: {reflection_result.issues}")'),
    ('print(f"[Pipeline] sample {retry_count}/{self.config.max_retries}, "\n                          f"sample: {failure_analysis.get(\'root_cause\', \'unknown\')}")',
     'print(f"[Pipeline] Retry {retry_count}/{self.config.max_retries}, "\n                          f"Cause: {failure_analysis.get(\'root_cause\', \'unknown\')}")'),
    ('print(f"[Pipeline] sample: {e}")', 'print(f"[Pipeline] Error: {e}")'),
    ('key_considerations=["sample"],', 'key_considerations=["Default fallback plan"],'),
    ('f"sample={pose.get(\'gripper_width\', 0.08):.3f}m, "',
     'f"gripper_width={pose.get(\'gripper_width\', 0.08):.3f}m, "'),
    ('print(f"[ExperienceEvolver] sample {len(experience_ids)} experiences (classEN: {failure_type})")',
     'print(f"[ExperienceEvolver] Penalized {len(experience_ids)} experiences (failure: {failure_type})")'),
    # Suggestion matching logic
    ('if "sample" in suggestion or "force" in suggestion_lower:\n                if "sample" in suggestion or "sample" in suggestion:\n                    adjustments["force_adjustment"] = "increase"\n                elif "sample" in suggestion or "sample" in suggestion:\n                    adjustments["force_adjustment"] = "decrease"',
     'if "force" in suggestion_lower or "strength" in suggestion_lower:\n                if "increase" in suggestion_lower or "firm" in suggestion_lower:\n                    adjustments["force_adjustment"] = "increase"\n                elif "decrease" in suggestion_lower or "gentle" in suggestion_lower:\n                    adjustments["force_adjustment"] = "decrease"'),
    ('if "sample" in suggestion or "side" in suggestion_lower:',
     'if "approach" in suggestion_lower or "side" in suggestion_lower:'),
    ('if "sample" in suggestion or "sample" in suggestion:\n                    adjustments["gripper_adjustment"] = 0.01\n                elif "sample" in suggestion or "sample" in suggestion:\n                    adjustments["gripper_adjustment"] = -0.01',
     'if "wider" in suggestion_lower or "open" in suggestion_lower:\n                    adjustments["gripper_adjustment"] = 0.01\n                elif "narrow" in suggestion_lower or "close" in suggestion_lower:\n                    adjustments["gripper_adjustment"] = -0.01'),
])

# --- context_builder.py ---
print("context_builder.py")
fix("core/context_builder.py", [
    ('f"- graspposition: {example[\'grasp_summary\']}"',
     'f"- Grasp summary: {example[\'grasp_summary\']}"'),
    ('"sample, outputENsceneENtarget_objectENgrasp_pose。"',
     '"Based on the above context, analyze the scene and target object, then output an optimal grasp pose."'),
    ('f"- successgrasp_pose:"', 'f"- Successful grasp pose:"'),
    ('f"- taskEN: {context[\'query\'][\'task\']}"', 'f"- Task: {context[\'query\'][\'task\']}"'),
    ('f"- sceneimage: {context[\'query\'][\'image_ref\']}"',
     'f"- Scene image: {context[\'query\'][\'image_ref\']}"'),
])

# --- embedding.py ---
print("embedding.py")
fix("core/embedding.py", [
    ('raise ImportError("sample open_clip_torch: pip install open_clip_torch")',
     'raise ImportError("open_clip_torch is required: pip install open_clip_torch")'),
    ('raise ImportError("sample sentence-transformers: pip install sentence-transformers")',
     'raise ImportError("sentence-transformers is required: pip install sentence-transformers")'),
])

# --- affordance_rag.py remaining ---
print("affordance_rag.py")
fix("core/affordance_rag.py", [
    ('"object_name": "objectEN"', '"object_name": "target_object"'),
    # The retrieval done line (slightly different format)
    ('print(f"[HAA-RAG] retrievaldone, time: {elapsed*1000:.1f}ms, returns {len(final_results)} sample")',
     'print(f"[HAA-RAG] Retrieval complete in {elapsed*1000:.1f}ms, returning {len(final_results)} results")'),
    ('f"{result.object_name}: {result.affordance_info.primary_affordance.value}, sample={result.final_score:.2f}"',
     'f"{result.object_name}: {result.affordance_info.primary_affordance.value}, score={result.final_score:.2f}"'),
])

# --- experience_learner.py remaining ---
print("experience_learner.py")
fix("agent/experience_learner.py", [
    # The object_name="sample" with 12-space indent (not 8)
    ('f"object\'{original_failure.object_name}\'failureclassEN\'{original_failure.failure_type}\'",',
     'f"Object \'{original_failure.object_name}\' failure type: \'{original_failure.failure_type}\'",'),
    ('lessons.append("RAGretrievalEN, experiencesEN")',
     'lessons.append("Leveraged RAG retrieval for experience-based learning")'),
    ('"collision": "planningEN",', '"collision": "Adjust planning path to avoid collision",'),
    ('failure_cause="objectEN",', 'failure_cause="Object slipped during test",'),
    ('failure_cause="objectEN",', 'failure_cause="Object contact lost",'),
    ('print(f"   experiencesEN: {stats[\'total_experiences\']}")',
     'print(f"   Total experiences: {stats[\'total_experiences\']}")'),
    ('print(f"   success: {stats[\'success_count\']}, failure: {stats[\'failure_count\']}, sample: {stats[\'corrected_count\']}")',
     'print(f"   Success: {stats[\'success_count\']}, Failure: {stats[\'failure_count\']}, Corrected: {stats[\'corrected_count\']}")'),
    ('print("\\n5. classEN:")', 'print("\\n5. Category rules:")'),
])

# --- agentic_grasp.py remaining ---
print("agentic_grasp.py")
fix("agent/agentic_grasp.py", [
    ('"object_name": "sample",\n            "category": "cup"',
     '"object_name": "ceramic_mug",\n            "category": "cup"'),
    ('"object_name": "sample",\n            "category": "tool"',
     '"object_name": "screwdriver",\n            "category": "tool"'),
    ('print(f"  - experiencesEN: {stats.get(\'total_experiences\', 0)}")',
     'print(f"  - Total experiences: {stats.get(\'total_experiences\', 0)}")'),
])

# --- react_engine.py remaining ---
print("react_engine.py")
fix("agent/react_engine.py", [
    ('print("ReAct inferenceEN")', 'print("ReAct Reasoning Trace")'),
])

# --- schema.py remaining ---
print("schema.py")
fix("knowledge_base/schema.py", [
    ('description="inferenceEN"', 'description="Reasoning explanation"'),
])

# --- grasp_memory.py remaining ---
print("grasp_memory.py")
fix("knowledge_base/grasp_memory.py", [
    # The description default is "A sample test object" - this was our own text, fine as-is since it's a test helper
    # Actually let's leave it, it's a test utility function
])

# --- benchmark.py remaining ---
print("benchmark.py")
fix("scripts/benchmark.py", [
    # The warmup lines use different context 
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")\n        for _ in range(warmup_runs)',
     'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")\n        for _ in range(warmup_runs)'),
    ('print(f"\\n[INFO] sample {warmup_runs} sample...")\n        for _ in range(warmup_runs)',
     'print(f"\\n[INFO] Running {warmup_runs} warmup iterations...")\n        for _ in range(warmup_runs)'),
    ('print("\\n| sample | sample |")', 'print("\\n| Metric | Value |")'),
    ('print(f"| sample | {all_results[\'embedding\'][\'visual_encoding\'][\'per_image_ms\']:.2f} ms/image |")',
     'print(f"| Visual encoding | {all_results[\'embedding\'][\'visual_encoding\'][\'per_image_ms\']:.2f} ms/image |")'),
    ('print(f"| sample | {all_results[\'embedding\'][\'text_encoding\'][\'per_text_ms\']:.2f} ms/text |")',
     'print(f"| Text encoding | {all_results[\'embedding\'][\'text_encoding\'][\'per_text_ms\']:.2f} ms/text |")'),
])

# --- experience_learner.py: object_name="sample" (at 12-space indent) ---
# Need to read exact lines
print("experience_learner.py object_names")
fix("agent/experience_learner.py", [
    # These have different indent levels  
    ('object_name="sample",\n            object_category="cup"', 'object_name="ceramic_cup",\n            object_category="cup"'),
    ('object_name="sample",\n            object_category="tool"', 'object_name="screwdriver",\n            object_category="tool"'),
    ('object_name="sample",\n            object_category="food"', 'object_name="banana",\n            object_category="food"'),
])

# --- agentic_grasp_pipeline.py: planningfailure ---
print("agentic_grasp_pipeline.py planningfailure fix")
fix("core/agentic_grasp_pipeline.py", [
    ('"planningfailure"', '"Planning failure"'),
])

print(f"\nTotal second-pass fixes: {total}")
