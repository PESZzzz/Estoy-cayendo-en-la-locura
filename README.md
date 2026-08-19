# PESZzzz-Modly-hunyuan3d-2.1

**English** | [Español](README.es.md)

## What is this?

Practically, a Modly extension for Hunyuan3D 2.1 designed for Windows AMD users and modest laptops or PCs.

It replaces and patches the internal code of Hunyuan3D 2.1 to run on hardware that Tencent never optimized for.

As I said, this is designed for Modly. It is **NOT** a standalone script. But, hey, the whole folder structure inside `hy3dshape` contains all the patches. If you want to wire it into ComfyUI or your own tool, go for it! (I don't recommend ComfyUI in any case, especially for AMD).

Testing showed it can work on several computers, even with NVIDIA. Albeit, it can also *explode*, so... take care.

Hey, are you still here? Great, that means you have the ability to read :D!

---

## What did I change?

Honestly, I *don't* remember. I'm **not** a programmer :D! I just... used AI.

But don't worry, you'll find all the explanations and changes right here:  
👉 **[Read the Technical Breakdown (summary.md)](summary.md)**

The main patched files are these:
- `pipelines.py`
- `surface_extractors.py`
- `volume_decoders.py`

---

## Requirements

Time for the *BEST* part:

| Component   | Minimum               | Recommended               |
| ----------- | --------------------- | ------------------------- |
| **OS**      | Windows 10/11, Linux  | Windows 11 / Ubuntu 22.04 |
| **RAM**     | **16 GB**             | 32 GB                     |
| **GPU**     | None required         | AMD / NVIDIA / Intel      |
| **CPU**     | AMD Ryzen 5           | AMD Ryzen 7               |
| **Free Space** | 15 GB              | 20 GB                     |
| **Modly**   | Latest stable         | Latest stable             |

> 📌 **Note:** This extension was built with **AMD** users in mind. If you have an **NVIDIA GPU**, you might prefer AlefK1708's [modly-hunyuan3d-21-lowvram](https://github.com/Alefk1708/modly-hunyuan3d-21-lowvram).

---

## Installation

1. Open Modly.
2. Go to **Extensions**.
3. Click **Install from GitHub**.
4. Paste the repository link: `https://github.com/PESZzzz/PESZzzz-modly-hunyuan3d-2.1`

That easy!

---

## Manual Installation (if you know what you are doing)

1. On GitHub, press **Code** and click **Download ZIP**.

...

Did you *expect* something "technical" like `git clone`? You can just... you know, download it.

---

## Usage in Modly

### Recommended Parameters

| Parameter           | Fast        | Balanced   | High Quality |
| ------------------- | ----------- | ---------- | ------------ |
| **Quality**         | 15 steps    | 30 steps   | 50 steps     |
| **Mesh Resolution** | 256         | 384        | 512          |
| **Guidance Scale**  | \-          | Any scale  | \-           |
| **Max Vertices**    | \-          | Depends on the model | \- |

**In more detail:**
- **Quality:** Determines the diffusion steps.
- **Mesh Resolution:** 256 is safe for 16 GB RAM. 384+ needs more RAM / VRAM.
- **Guidance Scale:** How closely the mesh follows the input image.
- **Max Vertices:** Decimates the output mesh to your vertex limit.
- **Seed:** Keep it at `-1` for random. Change it if you know what you are doing.

---

## IMPORTANT!!!

1. **Generation is very slow on CPU.**  
   What did you expect? On an 8-core laptop using 30 steps, it may take 4–5 hours or more. I *HIGHLY* recommend using 15 steps and a low Mesh Resolution for quick previews.

2. **Close ALL the apps you're using.**  
   The first few minutes of loading use a lot of RAM. As time goes on after loading, it will stabilize. You can even watch some YouTube videos while you wait!

3. **This project is still in development.**  
   If you find any issue, let me know on Twitter/X: [@SinJeshua](https://x.com/SinJeshua).

---

Finally, this project took a lot of work. I'm not a programmer; I'm just another student with a regular laptop, without experience or money. All these optimization ideas, like the "Hybrid Strategy," were made by me, not by AI. This was a full learning experience for me, and I learned a lot during this project. 

If you want to support my work, you can buy me a coffee on Ko-fi:  
☕ **[https://ko-fi.com/peszs](https://ko-fi.com/peszs)**

Any support helps me a lot!

*Fun fact: The AI tools I used to make this project made me cry. Sometimes, they're so dumb D:*

---

## FAQ

**Q: I have 16 GB of RAM, will my PC crash during load?**  
A: Absolutely not :D! As I said: close **all** the apps you're using. The first few minutes of loading use a lot of RAM. After loading finishes, memory usage stabilizes. You can even watch some YouTube videos while you wait!

**Q: What is the fastest setting to test if this works on my setup?**  
A: 15 steps, 256 Mesh Resolution, and 3 or 4 Guidance Scale. Try it first with a simple image.

**Q: Why does it say it can "explode" on NVIDIA cards?**  
A: It won't *literally* explode... or maybe yes :D! But seriously, this mod was tailored for AMD. If you have NVIDIA, AlefK1708's mod is way better optimized for CUDA.

**Q: Can I use this code outside of Modly?**  
A: Absolutely! The `hy3dshape` folder contains all the custom patches.

**Q: What am I supposed to do if you don't remember what you changed?**  
A: Don't worry about it! Check out the **[summary.md](summary.md)** file, or look inside the code files for comments tagged `[COMMUNITY]`.

**Q: Are you going to update it?**  
A: Maybe. Don't expect regular updates or fixes. I'm just a student!

**Q: Is there more to optimize?**  
A: Absolutely, yes. Maybe I can squeeze out more optimizations in the future.

---

## Licenses & Legal Notices

This project combines, modifies, and extends code from open-source technologies subject to their respective licenses:

* **Tencent Hunyuan 3D 2.1:** Released under the **TENCENT HUNYUAN 3D 2.1 COMMUNITY LICENSE AGREEMENT**. Please see the [`LICENSE`](LICENSE) file for the full legal terms and the Acceptable Use Policy (Exhibit A).
* **Modly Extension Base:** Released under the **MIT License** Copyright (c) 2026 Lightning Pixel.
* **Third-Party Components & Dependencies:** This distribution incorporates components and concepts from third-party works including **Stable Diffusion** (Stability AI) and **Flux** (Black Forest Labs).

> **Important Notice:** In compliance with upstream licenses, all third-party notices, credits, and open-source licenses are preserved in the [`NOTICE`](NOTICE) file located in the root directory.

Community patches and modifications introduced in this repository are released under the same terms as the upstream project.

[![License: Tencent](https://img.shields.io/badge/license-Tencent%20Hunyuan%20Community%20License-blue.svg)](LICENSE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
