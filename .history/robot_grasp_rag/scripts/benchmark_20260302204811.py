"""Core module implementation."""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw

# Note

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.grasp_agent import GraspAgent, GraspAgentConfig
from core.vlm_engine import VLMEngine, VLMConfig, GenerationConfig
from core.embedding import EmbeddingModel
from knowledge_base.vector_store import ChromaVectorStore, VectorStoreConfig


def create_test_images(n: int = 10) -> List[Image.Image]:


    """create_test_images function."""
    images = []
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    
    for i in range(n):
        img = Image.new("RGB", (640, 480), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        
        # Note

        draw.rectangle([0, 240, 640, 480], fill=(180, 160, 140))
        
        # Note

        color = colors[i % len(colors)]
        cx = 200 + (i * 50) % 300
        cy = 250 + (i * 20) % 100
        
        if i % 2 == 0:
            draw.ellipse([cx-40, cy-40, cx+40, cy+40], fill=color)
        else:
            draw.rectangle([cx-50, cy-30, cx+50, cy+30], fill=color)
            
        images.append(img)
        
    return images


def benchmark_vlm_inference(
    n_runs: int = 20,
    warmup_runs: int = 3,
) -> Dict[str, Any]:
    """benchmark_vlm_inference function."""
    print("\n" + "="*60)
    print("VLM inferencebenchmark")
    print("="*60)
    
    config = VLMConfig(
        model_path="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
    )
    
    engine = VLMEngine(config)
    engine.initialize()
    
    # Note

    test_image = create_test_images(1)[0]
    prompt = "Describe the objects in the scene and suggest grasp strategies."
    gen_config = GenerationConfig(max_tokens=200, temperature=0.1)
    
    # Note

    print(f"\n[INFO] Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = engine.generate(prompt, [test_image], gen_config)
        
    # Note

    print(f"[INFO] Running {n_runs} benchmark iterations...")
    latencies = []
    token_counts = []
    
    for i in range(n_runs):
        start = time.time()
        response = engine.generate(prompt, [test_image], gen_config)
        elapsed = time.time() - start
        
        latencies.append(elapsed)
        # Note

        token_counts.append(len(response) // 2)
        
        if (i + 1) % 5 == 0:
            print(f"  done {i+1}/{n_runs}")
            
    # Note

    import torch
    memory_gb = torch.cuda.memory_allocated() / 1024**3
    
    results = {
        "n_runs": n_runs,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000,
        "p90_latency_ms": np.percentile(latencies, 90) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "avg_tokens": np.mean(token_counts),
        "throughput_tps": sum(token_counts) / sum(latencies),
        "memory_gb": memory_gb,
    }
    
    print("\nresult:")
    print(f"  Avg latency: {results['avg_latency_ms']:.1f} +/- {results['std_latency_ms']:.1f} ms")
    print(f"  P50/P90/P99: {results['p50_latency_ms']:.1f} / {results['p90_latency_ms']:.1f} / {results['p99_latency_ms']:.1f} ms")
    print(f"  throughput: {results['throughput_tps']:.1f} tokens/s")
    print(f"  gpu_memory: {results['memory_gb']:.2f} GB")
    
    return results


def benchmark_embedding(
    n_images: int = 50,
    n_texts: int = 100,
) -> Dict[str, Any]:
    """benchmark_embedding function."""
    print("\n" + "="*60)
    print("benchmark")
    print("="*60)
    
    model = EmbeddingModel()
    
    # Note

    images = create_test_images(n_images)
    texts = [f"testEN {i}: benchmarkobjectsdescription" for i in range(n_texts)]
    
    # Note

    print(f"\n[INFO] Benchmarking ({n_images} images)...")
    start = time.time()
    visual_embeddings = model.encode_image(images)
    visual_time = time.time() - start
    
    # Note

    print(f"[INFO] Benchmarking ({n_texts} text queries)...")
    start = time.time()
    text_embeddings = model.encode_text(texts)
    text_time = time.time() - start
    
    results = {
        "visual_encoding": {
            "n_images": n_images,
            "total_time_ms": visual_time * 1000,
            "per_image_ms": visual_time * 1000 / n_images,
            "embedding_dim": visual_embeddings.shape[1],
        },
        "text_encoding": {
            "n_texts": n_texts,
            "total_time_ms": text_time * 1000,
            "per_text_ms": text_time * 1000 / n_texts,
            "embedding_dim": text_embeddings.shape[1],
        },
    }
    
    print("\nresult:")
    print(f"  Visual encoding: {results['visual_encoding']['per_image_ms']:.2f} ms/image")
    print(f"  Text encoding: {results['text_encoding']['per_text_ms']:.2f} ms/text")
    print(f"  Visual dim: {results['visual_encoding']['embedding_dim']}")
    print(f"  Text dim: {results['text_encoding']['embedding_dim']}")
    
    return results


def benchmark_retrieval(
    n_queries: int = 100,
    db_size: int = 1000,
) -> Dict[str, Any]:
    """benchmark_retrieval function."""
    print("\n" + "="*60)
    print("vector retrieval benchmark")
    print("="*60)
    
    # Note

    vector_store = ChromaVectorStore(VectorStoreConfig(
        persist_directory="./data/benchmark_db",
        collection_name="benchmark",
    ))
    vector_store.initialize()
    
    embedding_model = EmbeddingModel()
    
    # Note

    print(f"\n[INFO] Benchmarking ({db_size} entries)...")
    
    if vector_store.count() < db_size:
        # Note

        batch_size = 100
        for batch_start in range(0, db_size, batch_size):
            batch_end = min(batch_start + batch_size, db_size)
            
            ids = [f"test_{i}" for i in range(batch_start, batch_end)]
            texts = [f"Object description {i}: benchmark object {i}" for i in range(batch_start, batch_end)]
            embeddings = embedding_model.encode_text(texts)
            metadatas = [{"index": i, "category": f"cat_{i % 10}"} for i in range(batch_start, batch_end)]
            
            vector_store.add(ids, embeddings, metadatas, texts)
            
        print(f"  Database size: {db_size} entries")
        
    # Note

    print(f"[INFO] Executing {n_queries} retrieval queries...")
    
    query_texts = [f"queryobject {i}" for i in range(n_queries)]
    query_embeddings = embedding_model.encode_text(query_texts)
    
    latencies = []
    for i, embedding in enumerate(query_embeddings):
        start = time.time()
        results = vector_store.query_by_text(embedding, n_results=10)
        elapsed = time.time() - start
        latencies.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  done {i+1}/{n_queries}")
            
    results = {
        "n_queries": n_queries,
        "db_size": db_size,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "std_latency_ms": np.std(latencies) * 1000,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "qps": len(latencies) / sum(latencies),
    }
    
    print("\nresult:")
    print(f"  Database size: {db_size} entries")
    print(f"  Avg latency: {results['avg_latency_ms']:.2f} ms")
    print(f"  P50/P99: {results['p50_latency_ms']:.2f} / {results['p99_latency_ms']:.2f} ms")
    print(f"  QPS: {results['qps']:.1f}")
    
    return results


def benchmark_end_to_end(
    n_runs: int = 10,
    warmup_runs: int = 2,
) -> Dict[str, Any]:
    """benchmark_end_to_end function."""
    print("\n" + "="*60)
    print("End-to-end benchmark (RAG pipeline)")
    print("="*60)
    
    config = GraspAgentConfig(
        model_path="/home/fudan222/ct/Qwen3-VL/models/Qwen3-VL-8B",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        top_k=3,
    )
    
    agent = GraspAgent(config)
    
    # Note

    test_images = create_test_images(n_runs + warmup_runs)
    tasks = [
        ("objects", "Plan a grasp for this object"),
        ("objects", "Plan a grasp for this object"),
        ("objects", "Plan a grasp for this object"),
    ]
    
    # Note

    print(f"\n[INFO] Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        target, task = tasks[i % len(tasks)]
        _ = agent.plan_grasp(test_images[i], target, task, use_rag=True, verbose=False)
        
    # Note

    print(f"[INFO] Running {n_runs} benchmark iterations...")
    
    latencies = []
    retrieval_times = []
    inference_times = []
    confidences = []
    
    for i in range(n_runs):
        target, task = tasks[i % len(tasks)]
        
        start = time.time()
        prediction = agent.plan_grasp(
            test_images[warmup_runs + i],
            target,
            task,
            use_rag=True,
            verbose=False,
        )
        elapsed = time.time() - start
        
        latencies.append(elapsed)
        retrieval_times.append(prediction.retrieval_time_ms)
        inference_times.append(prediction.inference_time_ms)
        confidences.append(prediction.confidence)
        
        if (i + 1) % 5 == 0:
            print(f"  done {i+1}/{n_runs}")
            
    # Note

    stats = agent.get_memory_stats()
    
    results = {
        "n_runs": n_runs,
        "total_latency": {
            "avg_ms": np.mean(latencies) * 1000,
            "std_ms": np.std(latencies) * 1000,
            "p50_ms": np.percentile(latencies, 50) * 1000,
            "p99_ms": np.percentile(latencies, 99) * 1000,
        },
        "retrieval": {
            "avg_ms": np.mean(retrieval_times),
            "std_ms": np.std(retrieval_times),
        },
        "inference": {
            "avg_ms": np.mean(inference_times),
            "std_ms": np.std(inference_times),
        },
        "confidence": {
            "avg": np.mean(confidences),
            "std": np.std(confidences),
        },
        "memory": stats,
    }
    
    print("\nresult:")
    print(f"  Avg latency: {results['total_latency']['avg_ms']:.1f} +/- {results['total_latency']['std_ms']:.1f} ms")
    print(f"  - retrieval: {results['retrieval']['avg_ms']:.1f} ms")
    print(f"  - inference: {results['inference']['avg_ms']:.1f} ms")
    print(f"  Avg confidence: {results['confidence']['avg']:.2f}")
    print(f"  gpu_memory: {stats.get('allocated_gb', 0):.2f} GB")
    
    return results


def run_all_benchmarks() -> Dict[str, Any]:


    """run_all_benchmarks function."""
    print("="*60)
    print("RAG-VLM Benchmark Test")
    print("="*60)
    
    all_results = {}
    
    # Note

    all_results["vlm_inference"] = benchmark_vlm_inference(n_runs=10)
    
    # Note

    all_results["embedding"] = benchmark_embedding(n_images=20, n_texts=50)
    
    # Note

    all_results["retrieval"] = benchmark_retrieval(n_queries=50, db_size=500)
    
    # Note

    all_results["end_to_end"] = benchmark_end_to_end(n_runs=5)
    
    # Note

    print("\n" + "="*60)
    print("Benchmark test complete")
    print("="*60)
    
    print("\n| Metric | Value |")
    print("|------|-----|")
    print(f"| VLM inferencelatency | {all_results['vlm_inference']['avg_latency_ms']:.1f} ms |")
    print(f"| VLM throughput | {all_results['vlm_inference']['throughput_tps']:.1f} tokens/s |")
    print(f"| Visual encoding | {all_results['embedding']['visual_encoding']['per_image_ms']:.2f} ms/image |")
    print(f"| Text encoding | {all_results['embedding']['text_encoding']['per_text_ms']:.2f} ms/text |")
    print(f"| retrievallatency | {all_results['retrieval']['avg_latency_ms']:.2f} ms |")
    print(f"| Avg Latency | {all_results['end_to_end']['total_latency']['avg_ms']:.1f} ms |")
    print(f"| gpu_memory | {all_results['vlm_inference']['memory_gb']:.2f} GB |")
    
    # Note

    import json
    os.makedirs("./output", exist_ok=True)
    with open("./output/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
        
    print("\nresultsaved to ./output/benchmark_results.json")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-VLM Grasp Planning Benchmark")
    parser.add_argument("--test", type=str, choices=["vlm", "embedding", "retrieval", "e2e", "all"],
                        default="all", help="testclassEN")
    parser.add_argument("--n-runs", type=int, default=10, help="testEN")
    
    args = parser.parse_args()
    
    if args.test == "vlm":
        benchmark_vlm_inference(n_runs=args.n_runs)
    elif args.test == "embedding":
        benchmark_embedding()
    elif args.test == "retrieval":
        benchmark_retrieval(n_queries=args.n_runs)
    elif args.test == "e2e":
        benchmark_end_to_end(n_runs=args.n_runs)
    else:
        run_all_benchmarks()
