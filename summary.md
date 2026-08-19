## For big brain programmers

If you're a real programmer, not like me. Here you'll find the things you want to know:

1. Smart Device Manager (Auto-Hardware Detection)

Automatically scans your machine to detect NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS), or CPU-only environments.

2. Hybrid Strategy

Runs the heavy diffusion steps on your GPU for speed, but automatically offloads the VRAM-hungry 3D Mesh Extraction to system RAM / CPU. This is the *"secret sauce"* that allows 4–8 GB VRAM laptops to complete generations without crashing

3. Memory-Mapped Loading

Reads model weights directly into virtual memory without creating duplicate copies in RAM during startup. Saves several gigabytes of peak RAM on 16 GB machines

4. Anti-Memory-Bomb Patch

Removed heavy deepcopy functions during image conditioning.

5. Aggressive Garbage Collection

Runs multi-pass memory cleanups after every generation to prevent memory leaks in continuous sessions

6. Smart Core Allocation

Caps PyTorch thread usage to 75% of your available CPU cores instead of 100%. This keeps Windows responsive, prevents thermal throttling, and stops your entire system from freezing mid-generation

7. Windows torch.compile Guard

Automatically disables risky C++ compilation routines on Windows CPU builds that usually lead to instant application crashes

8. Adaptive Mesh Exporting

Automatically adjusts mesh chunk sizes and octree resolutions based on available system RAM. This maybe sounds weird because you select the octree resolutions in Modly, but trust me, it makes sense

9. Pre-allocated Memory Tensors

Replaced memory-heavy torch.cat() operations during 3D volume decoding with pre-allocated tensors. Eliminates 200% memory spikes that previously caused OOM crashes

10. Immediate Tensor Destruction

Forces explicit del of intermediate chunk queries, directional tensors, and matrix logits inside loops

11 Fast GPU-to-CPU Hand-off

Offloads 3D grid logits to system RAM and immediately purges GPU tensors *before* running CPU Marching Cubes algorithms

12. Hierarchical Octree Cleanup

Clears PyTorch caching pools (`free_memory()`) between multi-resolution refinement levels to survive high-resolution 3D exports
