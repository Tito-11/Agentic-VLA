"""Core module implementation."""

import os
import sys
import time
import argparse
from typing import Optional
from PIL import Image, ImageDraw

# Note

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.grasp_agent import GraspAgent, GraspAgentConfig
from utils.visualization import visualize_grasp_prediction


def create_demo_scene() -> Image.Image:


    """create_demo_scene function."""
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    
    # Note

    draw.rectangle([0, 240, 640, 480], fill=(180, 160, 140))
    
    # Note

    draw.ellipse([150, 220, 250, 280], fill=(255, 100, 100), outline=(150, 50, 50))
    draw.rectangle([145, 250, 155, 280], fill=(255, 100, 100))  # Note
    
    # Note

    draw.rectangle([350, 260, 450, 275], fill=(100, 100, 200))  # Note
    draw.rectangle([450, 265, 520, 270], fill=(180, 180, 180))  # Note
    
    # Note

    draw.arc([250, 300, 350, 400], start=0, end=180, fill=(255, 220, 50), width=20)
    
    return img


def run_demo(
    image_path: Optional[str] = None,
    target_object: str = "cup",
    task_description: str = "grasptarget_object",
    use_rag: bool = True,
    save_result: bool = True,
    output_dir: str = "./output",
):
    """run_demo function."""
    print("="*60)
    print("RAG-VLM grasp planning demo")
    print("="*60)
    
    # Note

    if image_path and os.path.exists(image_path):
        print(f"\n[INFO] load_image: {image_path}")
        scene_image = Image.open(image_path).convert("RGB")
    else:
        print("\n[INFO] use_demo_scene")
        scene_image = create_demo_scene()
        
    print(f"[INFO] target_object: {target_object}")
    print(f"[INFO] task_description: {task_description}")
    print(f"[INFO] use_rag: {use_rag}")
    
    # Note

    print("\n[INFO] initialize GraspAgent...")
    config = GraspAgentConfig(
        model_path="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        top_k=3,
        num_examples=3,
    )
    agent = GraspAgent(config)
    
    # Note

    print("\n[INFO] execute grasp planning...")
    start_time = time.time()
    
    prediction = agent.plan_grasp(
        scene_image=scene_image,
        target_object=target_object,
        task_description=task_description,
        use_rag=use_rag,
        verbose=True,
    )
    
    total_time = time.time() - start_time
    
    # Note

    print("\n" + "="*60)
    print("planning result")
    print("="*60)
    
    pose = prediction.grasp_pose
    print(f"\ngrasp_pose:")
    print(f"  position: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}) m")
    print(f"  pose: ({pose.orientation.qx:.3f}, {pose.orientation.qy:.3f}, "
          f"{pose.orientation.qz:.3f}, {pose.orientation.qw:.3f})")
    print(f"  gripper_width: {pose.gripper_width:.3f} m")
    
    print(f"\nconfidence: {prediction.confidence:.2f}")
    print(f"reasoning: {prediction.reasoning[:200]}..." if len(prediction.reasoning) > 200 else f"reasoning: {prediction.reasoning}")
    
    print(f"\nPerformance metrics:")
    print(f"  retrieval_time: {prediction.retrieval_time_ms:.1f} ms")
    print(f"  inference_time: {prediction.inference_time_ms:.1f} ms")
    print(f"  total_time: {total_time*1000:.1f} ms")
    
    if prediction.retrieved_experiences:
        print(f"\nretrieved_experiences: {', '.join(prediction.retrieved_experiences)}")
        
    # Note

    stats = agent.get_memory_stats()
    print(f"\nmemory_stats:")
    print(f"  Short-term memory: {stats.get('short_term_count', 0)} entries")
    print(f"  Long-term memory: {stats.get('long_term_count', 0)} entries")
    print(f"  GPU gpu_memory: {stats.get('allocated_gb', 0):.2f} GB")
    
    # Note

    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        
        # Note

        scene_path = os.path.join(output_dir, "scene.jpg")
        scene_image.save(scene_path)
        
        # Note

        import json
        result_path = os.path.join(output_dir, "result.json")
        result_data = {
            "target_object": target_object,
            "task_description": task_description,
            "grasp_pose": pose.to_dict(),
            "confidence": prediction.confidence,
            "reasoning": prediction.reasoning,
            "retrieval_time_ms": prediction.retrieval_time_ms,
            "inference_time_ms": prediction.inference_time_ms,
            "retrieved_experiences": prediction.retrieved_experiences,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
            
        print(f"\nresultsaved to {output_dir}")
        
    print("\n" + "="*60)
    print("demo done!")
    print("="*60)
    
    return prediction


def run_batch_demo():


    """run_batch_demo function."""
    tasks = [
        ("cup", "Grasp the cup carefully"),
        ("screwdriver", "Pick up the screwdriver"),
        ("bottle", "Plan a grasp for this bottle"),
    ]
    
    print("="*60)
    print("batchgrasp planning demo")
    print("="*60)
    
    scene_image = create_demo_scene()
    
    config = GraspAgentConfig(
        model_path="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
    )
    agent = GraspAgent(config)
    
    results = []
    for target, task in tasks:
        print(f"\n--- task: {target} ---")
        prediction = agent.plan_grasp(
            scene_image=scene_image,
            target_object=target,
            task_description=task,
            use_rag=True,
            verbose=True,
        )
        results.append((target, prediction))
        
    # Note

    print("\n" + "="*60)
    print("batchresults")
    print("="*60)
    
    for target, pred in results:
        print(f"\n{target}:")
        print(f"  confidence: {pred.confidence:.2f}")
        print(f"  total_time: {pred.retrieval_time_ms + pred.inference_time_ms:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="grasp planning demo")
    parser.add_argument("--image", type=str, help="inputimagepath")
    parser.add_argument("--target", type=str, default="cup", help="Target object name")
    parser.add_argument("--task", type=str, default="grasptarget_object", help="task_description")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Run batch demo")
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch_demo()
    else:
        run_demo(
            image_path=args.image,
            target_object=args.target,
            task_description=args.task,
            use_rag=not args.no_rag,
            save_result=not args.no_save,
            output_dir=args.output,
        )
