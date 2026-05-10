"""Comprehensive fix for all EN/sample placeholder text across the codebase.
This script replaces broken placeholder text with proper English."""

import os
import re

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"

# === File-specific replacements ===
# Format: {filepath: [(old_string, new_string), ...]}

REPLACEMENTS = {
    # ---- agent/react_engine.py ----
    "agent/react_engine.py": [
        # Tool descriptions
        ('"ENknowledge_baseENretrievalobjectsENsuccessgraspexperience"',
         '"Retrieve successful grasp experiences for similar objects from the knowledge base"'),
        ('"querydescription, objectsENclassEN"',
         '"Query description, e.g. object name or category"'),
        ('"returnsEN (EN3)"',
         '"Number of results to return (default: 3)"'),
        ('"ENVLMENtarget_objectEN（sample、sample、sample、sample）"',
         '"Use VLM to analyze the target object properties (shape, material, size, graspable regions)"'),
        ('"objectEN"', '"object_name"'),
        ('"sample (sample)"', '"aspects to focus on (e.g. shape, material)"'),
        # AnalyzeObjectTool execute
        ('if self._vlm_engine is None:  # Note',
         'if self._vlm_engine is None:'),
        ('prompt = f"""imagesEN{object_name}, sample: 1. ENclassEN (sample/sample/sample/sample/sample)\n2. sample (sample/sample/sample/sample)\n3. sample (ENxENxEN cm)\n4. ENposition (sample/sample/sample/sample)\n5. sample (sample/sample/sample)\n6. ENgrasp sample\n\nENJSONENreturns。"""',
         'prompt = f"""Analyze the object \\"{object_name}\\" in the image and provide:\n1. Category (e.g. kitchenware/tool/stationery/container/food)\n2. Material (e.g. metal/plastic/glass/ceramic)\n3. Approximate dimensions (L x W x H in cm)\n4. Stable position (e.g. upright/lying/tilted/stacked)\n5. Surface properties (e.g. smooth/rough/textured)\n6. Recommended grasp strategy\n\nReturn your analysis in JSON format."""'),
        # PlanGraspTool
        ('"objectsENretrievalexperience, ENgrasp_poseargs"',
         '"Combine object analysis and retrieved experiences to plan an optimal grasp pose"'),
        # ExecuteGraspTool  
        ('"ENexecutegrasp sample, ENposeEN"',
         '"Execute the grasp action and verify the planned pose against constraints"'),
        # FailureAnalysisTool
        ('"ENgraspfailureEN, sample"',
         '"Analyze grasp failure causes and suggest improvements"'),
        # Failure analysis strings
        ('"ENpositionEN"', '"Position error detected"'),
        ('"ENgrasp"', '"Insufficient grasp stability"'),
        ('"ENgrasp sample"', '"Adjust grasp strategy"'),
        ('"ENposition"', '"Refine position estimate"'),
        ('"ENutilsEN"', '"Check tool constraints"'),
        ('"ENgripper_width (±15%)"', '"Adjust gripper width (±15%)"'),
        ('"ENfailureEN"', '"Unidentified failure cause"'),
        ('analysis["suggestions"].insert(0, "ENgrasp sample")',
         'analysis["suggestions"].insert(0, "Try alternative grasp approach")'),
        # ReAct prompt template
        ('REACT_PROMPT = """ENgraspplanningAgent, ENReActENinferenceEN。\n\nEN:\n{task_description}\n\nENutils:\n{tools_prompt}\n\nENinference: Thought: sample\nAction: ENutilsEN\nAction Input: {{"param": "value"}}\nObservation: ENresult\n\n... (ENThought/Action/ObservationENdone)\nThought: ENdonetask\nFinal Answer: ENgraspplanning result\n\nENinference:\n{react_history}\n\nENinference (ENtaskdone, ENFinal Answer):"""',
         'REACT_PROMPT = """You are a robotic grasp planning agent. Use the ReAct (Reasoning + Acting) framework to solve the task step by step.\n\nTask:\n{task_description}\n\nAvailable Tools:\n{tools_prompt}\n\nReasoning Format:\nThought: Describe what you are thinking\nAction: Choose a tool to use\nAction Input: {{"param": "value"}}\nObservation: Tool execution result\n\n... (Repeat Thought/Action/Observation until the task is complete)\nThought: I have gathered enough information to complete the task.\nFinal Answer: The complete grasp plan.\n\nReasoning so far:\n{react_history}\n\nContinue reasoning (complete the task and provide a Final Answer):"""'),
        # Simulated responses
        ('return """Thought: ENplanninggrasp',
         'return """Thought: Planning grasp strategy'),
        ('"reasoning": "objectsexperiences, ENgrasp"',
         '"reasoning": "Based on object analysis and retrieved experiences, planning optimal grasp"'),
        ('return """Thought: ENretrievalobjectsENgraspexperience',
         'return """Thought: Retrieving similar object grasp experiences'),
        ('Action Input: {"query": "ENgrasp", "top_k": 3}"""',
         'Action Input: {"query": "grasp similar object", "top_k": 3}"""'),
        ('return """Thought: ENresultplanninggrasp_pose',
         'return """Thought: Using retrieved results to plan grasp pose'),
        ('"reference_experiences": "ENgraspexperience"}"""',
         '"reference_experiences": "Retrieved grasp experiences applied"}"""'),
        ('return """Thought: ENplanningEN',
         'return """Thought: Finalizing the grasp plan'),
        ('"grasp_plan": "ENgrasp", "object_properties": "sample"}"""',
         '"grasp_plan": "Optimized grasp", "object_properties": "Analyzed"}"""'),
        # Misc
        ('"ENpath"', '"Approach path"'),
        ('f"objectENwidth({actual_width:.2f}m)ENplanning({planned_width:.2f}m)sample"',
         'f"Object width ({actual_width:.2f}m) differs from planned ({planned_width:.2f}m)"'),
    ],

    # ---- agent/reflection.py ----
    "agent/reflection.py": [
        ('"ENpathplanningEN"', '"Approach path planning error"'),
        ('"ENgripper_width"', '"Incorrect gripper width"'),
        ('"ENgrasp sample"', '"Unstable grasp configuration"'),
        ('"ENgrasp"', '"Adjust grasp strategy"'),
        ('"ENposition"', '"Refine position estimate"'),
        ('"ENgrasp sample"', '"Try alternative grasp pose"'),
        ('"ENutilsEN"', '"Verify tool constraints"'),
        ('alternatives.insert(0, "ENgrasp sample")', 'alternatives.insert(0, "Try a wider grasp approach")'),
        # Reflection prompt
        ('prompt = f"""ENgraspfailureEN: object: {context.object_name} ({context.object_category})',
         'prompt = f"""Analyze the following grasp failure: object: {context.object_name} ({context.object_category})'),
        ('1. ENfailureEN', '1. Root cause of the failure'),
    ],

    # ---- agent/agentic_grasp.py ----
    "agent/agentic_grasp.py": [
        ('print(f"[Agent] ENargs: {json.dumps(initial_params, ensure_ascii=False, indent=2)}")',
         'print(f"[Agent] Initial params: {json.dumps(initial_params, ensure_ascii=False, indent=2)}")'),
        ('print("[Agent] sample, ENargs...")',
         'print("[Agent] Retrying with adjusted params...")'),
        ('print(f"\\n[Agent] sample {attempt_num} ENexecute...")',
         'print(f"\\n[Agent] Attempt {attempt_num} executing...")'),
        ('print(f"[Agent] ENargs: {json.dumps(current_params, ensure_ascii=False)}")',
         'print(f"[Agent] Adjusted params: {json.dumps(current_params, ensure_ascii=False)}")'),
        ('"task": "ENgrasp sample"', '"task": "Grasp the target object"'),
        ('print(f"ENtime: {avg_time:.0f}ms")', 'print(f"Average time: {avg_time:.0f}ms")'),
    ],

    # ---- agent/experience_learner.py ----
    "agent/experience_learner.py": [
        ('print("\\n1. ENsuccessexperience...")', 'print("\\n1. Storing successful experience...")'),
        ('print("\\n2. ENfailureexperience...")', 'print("\\n2. Storing failure experience...")'),
        ('print("\\n3. ENsuccessexperience...")', 'print("\\n3. Storing another successful experience...")'),
    ],

    # ---- agent/grasp_agent.py ----
    "agent/grasp_agent.py": [
        ('print(f"[GraspAgent] ENfailure: {e}")', 'print(f"[GraspAgent] Execution failed: {e}")'),
    ],

    # ---- agent/retry_strategy.py ----
    "agent/retry_strategy.py": [
        ('return False, f"ENsuccessEN({success_prob:.1%})sample"',
         'return False, f"Low success probability ({success_prob:.1%}), skipping retry"'),
        ('print(f"[Retry] ENsuccessEN: {plan[\'success_probability\']:.1%}")',
         'print(f"[Retry] Estimated success probability: {plan[\'success_probability\']:.1%}")'),
        ('print(f"[Retry] ✓ sample {attempt} ENsuccess!")',
         'print(f"[Retry] ✓ Attempt {attempt} succeeded!")'),
        ('print(f"[Retry] ✗ ENfailure: {failure_type.value if failure_type else \'unknown\'}")',
         'print(f"[Retry] ✗ Failed: {failure_type.value if failure_type else \'unknown\'}")'),
        ('print("ENresults")', 'print("Results:")'),
        ('print(f"ENsuccess: {result.final_success}")', 'print(f"Success: {result.final_success}")'),
    ],

    # ---- core/affordance_rag.py ----
    "core/affordance_rag.py": [
        # Prediction prompt
        ('PREDICTION_PROMPT = """ENtarget_object, ENgrasp sample(Affordance)sample。\n\nENoutput JSON sample: ```json',
         'PREDICTION_PROMPT = """Analyze the target object and predict its affordance properties for grasp planning.\n\nOutput in JSON format: ```json'),
        ('"primary_affordance": "ENclassEN"',
         '"primary_affordance": "affordance category"'),
        ('"graspable_regions": ["ENgrasp sample"]',
         '"graspable_regions": ["list of graspable regions"]'),
        ('"reasoning": "ENinference"',
         '"reasoning": "your reasoning"'),
        ('ENclassEN: - graspable_handle: sample',
         'Affordance categories:\n- graspable_handle: objects with handles'),
        ('- wrappable: ENgrasp',
         '- wrappable: objects that can be wrapped/enveloped'),
        ('ENoutput JSON, sample。"""',
         'Output valid JSON only."""'),
        ('print(f"[AffordancePredictor] VLM ENfailure: {e}")',
         'print(f"[AffordancePredictor] VLM prediction failed: {e}")'),
        ('print(f"[AffordancePredictor] ENfailure: {e}")',
         'print(f"[AffordancePredictor] Prediction failed: {e}")'),
        ('print("[HAA-RAG] ENretrievalENresult")',
         'print("[HAA-RAG] No retrieval results found")'),
        ('print("[HAA-RAG] ENretrievalENresult, ENresult")',
         'print("[HAA-RAG] Retrieval complete, processing results")'),
        ('return "ENposeEN"', 'return "Default grasp pose"'),
        # Build prompt
        ('return f"""ENgraspplanningEN, objectsEN"sample"(Affordance)ENgraspplanning。',
         'return f"""Plan the grasp based on the object\'s affordance properties for optimal grasping.'),
        # Graspable regions
        ("', '.join(affordance.graspable_regions) if affordance.graspable_regions else 'sample'",
         "', '.join(affordance.graspable_regions) if affordance.graspable_regions else 'not specified'"),
        # Affordance strategy strings
        ('AffordanceType.GRASPABLE_HANDLE: "ENgrasp sample, sample(pinch)sample, sample"',
         'AffordanceType.GRASPABLE_HANDLE: "Grasp the handle with a pinch grip for optimal control"'),
        ('AffordanceType.WRAPPABLE: "ENgrasp, sample, sample"',
         'AffordanceType.WRAPPABLE: "Use an enveloping grasp for maximum contact area"'),
        ('AffordanceType.DEFORMABLE: "ENgrasp, sample"',
         'AffordanceType.DEFORMABLE: "Apply gentle force with adaptive grip"'),
        ('base_strategy += "\\n⚠️ objectEN, sample,, ENgripper_width"',
         'base_strategy += "\\n⚠️ Fragile object detected, reduce gripper force"'),
        ('base_strategy += "\\n💪 objectEN, ENgrasp sample"',
         'base_strategy += "\\n💪 Heavy object detected, use firm grasp"'),
    ],

    # ---- core/agentic_grasp_pipeline.py ----
    "core/agentic_grasp_pipeline.py": [
        # Reflection prompt
        ('REFLECTION_PROMPT = """ENgrasp sample。ENgrasp sample, sample。',
         'REFLECTION_PROMPT = """Review the grasp execution result. Analyze the grasp plan and suggest improvements.'),
        ('- ENclassEN: {affordance_type}',
         '- Affordance type: {affordance_type}'),
        ('"issues": ["EN1", "EN2"]', '"issues": ["issue 1", "issue 2"]'),
        ('"suggestions": ["EN1", "EN2"]', '"suggestions": ["suggestion 1", "suggestion 2"]'),
        ('"retry_strategy": "ENdescription"', '"retry_strategy": "description of retry strategy"'),
        ('print(f"[Reflection] ENfailure: {e}")', 'print(f"[Reflection] Analysis failed: {e}")'),
        ('issues=[f"ENmoduleEN: {str(e)}"]',
         'issues=[f"Reflection module error: {str(e)}"]'),
        ('issues=["ENresult"]', 'issues=["Unable to parse result"]'),
        # Failure analysis prompt
        ('ANALYSIS_PROMPT = """ENgraspfailureEN, sample。',
         'ANALYSIS_PROMPT = """Analyze the grasp failure and identify root causes.'),
        ('- ENposition: ({px:.3f}, {py:.3f}, {pz:.3f})',
         '- Position: ({px:.3f}, {py:.3f}, {pz:.3f})'),
        ('ENfailureENoutput JSON:', 'Analyze the failure and output JSON:'),
        ('"root_cause": f"ENfailure: {str(e)}"',
         '"root_cause": f"Analysis error: {str(e)}"'),
        ('"root_cause": "ENresult"', '"root_cause": "Unable to determine root cause"'),
        # Planning prompt
        ('PLANNING_PROMPT = """ENgraspplanningEN。ENsceneimageENretrieved_experiences, ENgrasp sample。',
         'PLANNING_PROMPT = """Plan the grasp action. Use the scene image and retrieved experiences to generate an optimal grasp pose.'),
        ('ENgrasp sample, output JSON:', 'Generate a grasp plan and output JSON:'),
        ('"key_considerations": ["EN1", "EN2"]', '"key_considerations": ["consideration 1", "consideration 2"]'),
        ('return "ENretrieved_experiences"', 'return "No relevant experiences found"'),
        ('key_considerations=["ENfailure, sample"]',
         'key_considerations=["Fallback plan, limited information available"]'),
        ('"description": "ENexecutesuccess" if success else "ENexecutefailure"',
         '"description": "Execution succeeded" if success else "Execution failed"'),
    ],

    # ---- core/context_builder.py ----
    "core/context_builder.py": [
        ('return """ENgraspplanningEN。ENtaskENsceneENsuccessexperience, ENtarget_objectEN6Dgrasp_pose。',
         'return """You are a robotic grasp planning expert. Your task is to analyze the scene and past successful experiences to generate a 6D grasp pose for the target object.'),
        ('"reasoning": string   // ENinferenceEN',
         '"reasoning": string   // Brief explanation of the grasp strategy'),
        ('1. ENsuccessexperience, objectsEN',
         '1. Study successful experiences, especially for similar objects'),
        ('2. ENgrasp sample（sample、objectEN）',
         '2. Evaluate grasp feasibility (shape, material, object properties)'),
        ('4. ENgrasp_poseEN"""',
         '4. Generate a complete and precise grasp pose."""'),
        ('"image_ref": "[ENsceneimage]"', '"image_ref": "[Scene image]"'),
        ('parts.append("## ENsuccessexperience")', 'parts.append("## Successful Grasp Experiences")'),
        ('parts.append("## ENtask")', 'parts.append("## Current Task")'),
        ('print(f"[ContextBuilder] load_imagefailure: {e}")',
         'print(f"[ContextBuilder] Failed to load image: {e}")'),
    ],

    # ---- core/embedding.py ----
    "core/embedding.py": [
        ('print(f"[VisionEncoder] ENloaddone")', 'print(f"[VisionEncoder] Model loaded successfully")'),
        ('print(f"[TextEncoder] ENloaddone")', 'print(f"[TextEncoder] Model loaded successfully")'),
    ],

    # ---- core/multimodal_rag.py ----
    "core/multimodal_rag.py": [
        ('print(f"[MultiModalRetriever] retrievaldone: {len(results)} ENresult, {elapsed:.1f}ms")',
         'print(f"[MultiModalRetriever] Retrieval complete: {len(results)} results, {elapsed:.1f}ms")'),
    ],

    # ---- core/retriever.py ----
    "core/retriever.py": [
        ('print("[DualPathRetriever] ENretrievalENresult")',
         'print("[DualPathRetriever] No retrieval results found")'),
        ('print(f"[DualPathRetriever] retrievaldone, time: {elapsed*1000:.1f}ms, returns {len(filtered_results[:top_k])} E',
         'print(f"[DualPathRetriever] Retrieval complete, time: {elapsed*1000:.1f}ms, returns {len(filtered_results[:top_k])}'),
        ('print(f"[DualPathRetriever] classEN \'{category}\' ENresult")',
         'print(f"[DualPathRetriever] Category \'{category}\': no results found")'),
    ],

    # ---- core/scene_graph_rag.py ----
    "core/scene_graph_rag.py": [
        ('ENoutput JSON sample:', 'Output in JSON format:'),
        ('ENclassEN: on_top_of, under, inside, contains, next_to, left_of, right_of, in_front_of, behind, occludes, supports, near',
         'Supported relation types: on_top_of, under, inside, contains, next_to, left_of, right_of, in_front_of, behind, occludes, supports, near'),
        ('print(f"[SceneGraphBuilder] ENfailure: {e}")',
         'print(f"[SceneGraphBuilder] Build failed: {e}")'),
    ],

    # ---- core/unified_agentic_rag.py ----
    "core/unified_agentic_rag.py": [
        ('return "ENgrasp sample"', 'return "Apply standard grasp strategy"'),
    ],

    # ---- core/vlm_engine.py ----
    "core/vlm_engine.py": [
        ('print(f"[VLMEngine] ENloaddone, time: {load_time:.2f}s")',
         'print(f"[VLMEngine] Model loaded, time: {load_time:.2f}s")'),
        ('print("[VLMEngine] ENinference...")', 'print("[VLMEngine] Running inference...")'),
        ('print("[VLMEngine] ENdone")', 'print("[VLMEngine] Inference complete")'),
    ],

    # ---- knowledge_base/grasp_memory.py ----
    "knowledge_base/grasp_memory.py": [
        ('print(f"[GraspMemory] ENfailure: {e}")', 'print(f"[GraspMemory] Operation failed: {e}")'),
    ],

    # ---- knowledge_base/schema.py ----
    "knowledge_base/schema.py": [
        ('description="ENvector"', 'description="Embedding vector"'),
        ('description="ENpath"', 'description="File path"'),
        ('description="ENsuccess"', 'description="Whether the grasp was successful"'),
        ('description="ENgrasp_pose"', 'description="The 6D grasp pose"'),
    ],

    # ---- scripts/benchmark.py ----
    "scripts/benchmark.py": [
        ('prompt = "ENdescriptionobjects, ENgrasp sample。"',
         'prompt = "Describe the objects in the scene and suggest grasp strategies."'),
        ('print(f"  ENlatency: {results[\'avg_latency_ms\']:.1f} ± {results[\'std_latency_ms\']:.1f} ms")',
         'print(f"  Avg latency: {results[\'avg_latency_ms\']:.1f} +/- {results[\'std_latency_ms\']:.1f} ms")'),
        ('print(f"[INFO] execute {n_queries} ENretrieval...")',
         'print(f"[INFO] Executing {n_queries} retrieval queries...")'),
        ('print(f"  ENlatency: {results[\'avg_latency_ms\']:.2f} ms")',
         'print(f"  Avg latency: {results[\'avg_latency_ms\']:.2f} ms")'),
        ('print(f"  ENlatency: {results[\'total_latency\'][\'avg_ms\']:.1f} ± {results[\'total_latency\'][\'std_ms\']:.1f} ms")',
         'print(f"  Avg latency: {results[\'total_latency\'][\'avg_ms\']:.1f} +/- {results[\'total_latency\'][\'std_ms\']:.1f} ms")'),
        ('print(f"  ENconfidence: {results[\'confidence\'][\'avg\']:.2f}")',
         'print(f"  Avg confidence: {results[\'confidence\'][\'avg\']:.2f}")'),
        ('print("RAG-VLM ENbenchmarktest")', 'print("RAG-VLM Benchmark Test")'),
        ('print("ENbenchmarktestEN")', 'print("Benchmark test complete")'),
        ('print(f"| ENlatency | {all_results[\'end_to_end\'][\'total_latency\'][\'avg_ms\']:.1f} ms |")',
         'print(f"| Avg Latency | {all_results[\'end_to_end\'][\'total_latency\'][\'avg_ms\']:.1f} ms |")'),
        ('parser = argparse.ArgumentParser(description="ENbenchmarktest")',
         'parser = argparse.ArgumentParser(description="RAG-VLM Grasp Planning Benchmark")'),
    ],

    # ---- scripts/build_knowledge_base.py ----
    "scripts/build_knowledge_base.py": [
        ('"key_insight": "ENgrasp sample"', '"key_insight": "Standard grasp approach"'),
        ('"key_insight": "ENgrasp"', '"key_insight": "Optimal grasp strategy"'),
        ('"key_insight": "ENgrasp, sample"', '"key_insight": "Adapted grasp with special handling"'),
        ('print("ENgraspexperienceknowledge_base")', 'print("Building grasp experience knowledge base")'),
        ('print(f"- ENpath: {db_dir}")', 'print(f"- Database path: {db_dir}")'),
        ('parser = argparse.ArgumentParser(description="ENgraspexperienceknowledge_base")',
         'parser = argparse.ArgumentParser(description="Build grasp experience knowledge base")'),
    ],

    # ---- scripts/demo_grasp.py ----
    "scripts/demo_grasp.py": [
        ('"ENgrasp sample"', '"Plan a grasp for this object"'),
        ('"ENuse_rag"', '"Disable RAG retrieval"'),
        ('"ENsaveresult"', '"Do not save results"'),
    ],
}


def apply_replacements():
    total_fixed = 0
    for rel_path, pairs in REPLACEMENTS.items():
        filepath = os.path.join(ROOT, rel_path.replace("/", os.sep))
        if not os.path.exists(filepath):
            print(f"SKIP (not found): {rel_path}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        file_fixes = 0
        for old, new in pairs:
            if old in content:
                content = content.replace(old, new, 1)
                file_fixes += 1

        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {file_fixes} strings in {rel_path}")
            total_fixed += file_fixes
        else:
            # Try to find near-matches for debugging
            unfixed = []
            for old, new in pairs:
                if old not in original:
                    unfixed.append(old[:60])
            if unfixed:
                print(f"WARN {rel_path}: {len(unfixed)} replacements not matched")
                for u in unfixed[:3]:
                    print(f"  - {u}")

    print(f"\nTotal: {total_fixed} strings fixed")


if __name__ == "__main__":
    apply_replacements()
