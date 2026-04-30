# Chaos Hearing

> At the intersection of sound physics, auditory cognition, and interface design.

---

Chaos Hearing is a research project exploring how we hear, how sound behaves, and how we might build better tools for interacting with complex auditory environments. It sits at the crossroads of three threads:

1. **Physics of Sound** — nonlinear dynamics, chaotic systems, physical modeling of sound synthesis, and probabilistic methods for audio analysis
2. **Cognition of Hearing** — auditory scene analysis, musical imagery (hearing songs in your head), perfect pitch, and auditory phenomena in psychosis
3. **Interfaces for Interaction** — intuitive audio control in multi-source environments, drawing from HCI and computational auditory scene analysis (CASA)

The project grew out of a thesis on _Intuitive Audio Interaction and Control in Multi-Source Environments_ and a collaboration with [Myunghyun Oh](https://mathematics.ku.edu/myunghyun-oh) on the mathematics of hearing and chaotic dynamics.

## Repository Structure

```
chaoshearing/
├── README.md
├── index.md                       # GitHub Pages homepage
├── casa/                          # Computational Auditory Scene Analysis library
│   ├── __init__.py                # Package API
│   ├── kernels.py                 # Spectral mixture GPs & probabilistic TF analysis
│   ├── kalman.py                  # EP via Kalman smoothing (state-space GP, O(N))
│   ├── graphs.py                  # Similarity graphs & spectral geometry
│   ├── diffusion.py               # Absorbing-state discrete diffusion
│   ├── oscillators.py             # Hopf bifurcation cochlear model & filterbank
│   └── README.md                  # Module documentation
├── sources/                       # Research papers and annotated bibliography
│   ├── BIBLIOGRAPHY.md
│   └── *.pdf
├── projects/
│   ├── 01-listening-and-duets.md  # Cochlear dynamics & scene analysis
│   ├── 02-spectral-cognition.md   # Sound physics & auditory imagery
│   └── 03-shape-grammar.md        # Cross-layer shape grammar
├── grammar/                       # Formal shape grammar specification
│   ├── primitives.yaml            # Shape vocabulary (tiles, oscillators, illusions)
│   ├── rules.yaml                 # Production rules & cross-layer mappings
│   ├── isomorphisms.yaml          # Structural isomorphisms across layers
│   └── visualize.py               # Grammar → diagram renderer
├── notebooks/
│   ├── nonstationary_audio.py     # Spectral mixture GP demo
│   └── ep_kalman_viz.py           # EP via Kalman smoothing (Algorithm 1)
└── _config.yml
```

## `casa` — Computational Auditory Scene Analysis Library

The `casa/` package provides composable signal processing primitives for probabilistic auditory analysis:

```python
from casa import gp_regression, ep_kalman_smoother, cochlear_filterbank
from casa import build_similarity_graph, graph_laplacian, spectral_analysis
from casa import AbsorbingDiffusion
```

| Module        | What it does                                                          | Complexity |
| :------------ | :-------------------------------------------------------------------- | :--------- |
| `kernels`     | Spectral mixture GP inference — probabilistic time-frequency analysis | O(N³)      |
| `kalman`      | EP via Kalman smoothing — state-space GP for long signals             | O(N)       |
| `graphs`      | Similarity graphs & Laplacian spectral analysis                       | O(N² k)    |
| `diffusion`   | Absorbing-state discrete diffusion for structured sequences           | O(T·L)     |
| `oscillators` | Hopf bifurcation cochlear model & tonotopic filterbank                | O(C·T)     |

These compose into a full CASA pipeline: signal → transduction → spectral decomposition → temporal tracking → scene structure → generative reconstruction.

See [`casa/README.md`](casa/README.md) for detailed API documentation.

## Projects

### [01 — On Listening and Duets](https://intelligent-interfaces.github.io/chaoshearing/projects/01-listening-and-duets)

_Hearing Performance, Chaotic Dynamics, and Computational Auditory Scene Analysis_

The original thread: modeling the cochlea and auditory perception through the lens of nonlinear dynamics (Hopf bifurcations, limit cycles) and connecting that to how we parse complex sound scenes — duets, ensembles, overlapping conversations.

### [02 — Spectral Cognition](https://intelligent-interfaces.github.io/chaoshearing/projects/02-spectral-cognition)

_Sound Physics, Musical Imagery, and the Perception of Pitch_

Bridging the physics of sound (synthesis, physical modeling, probabilistic spectral analysis) with cognitive phenomena — perfect pitch, involuntary musical imagery, auditory hallucinations, and what these tell us about how the brain constructs sound.

### [03 — Shape Grammar](https://intelligent-interfaces.github.io/chaoshearing/projects/03-shape-grammar)

_A Layered Grammar Connecting Tile-Based GPU Computing, Auditory Science, and Perceptual Demos_

A shape grammar that operates across three layers of a perceptual computing stack:

| Layer         | Project                                                         | Role                                                                      |
| :------------ | :-------------------------------------------------------------- | :------------------------------------------------------------------------ |
| **Backend**   | [TileGym](https://github.com/NVIDIA/TileGym)                    | Computational substrate — CUDA tile kernels for GPU programming           |
| **Interface** | Chaos Hearing                                                   | Scientific model — nonlinear dynamics, spectral cognition, scene analysis |
| **Frontend**  | [McDermott Lab Demos](https://mcdermottlab.mit.edu/sounds.html) | Perceptual surface — auditory illusions and interactive demonstrations    |

The grammar identifies structural isomorphisms across all three layers — the same patterns of **decomposition, dispatch, fusion, and reduction** recur from silicon to synapse to subjective experience. Seven worked examples trace each isomorphism through concrete code (TileGym's `@dispatch`, `splitk_reduce`, `PartiallyFusedSwiGLUMLP`, attention sinks, MoE routing), equations (Hopf bifurcation, spectral mixture kernels, EP via Kalman smoothing), and perceptual phenomena (cocktail party segregation, illusory continuity, texture perception, perceptual bistability).

Extends into **causal uncertainty in sound objects** (manipulating the ambiguity of auditory causal inference) and **computational psychiatry** (perceptual disorders as production rule failures in the grammar).

See the full document: [`SHAPE_GRAMMAR.md`](../SHAPE_GRAMMAR.md)

## Grammar Specification

The `grammar/` directory contains a machine-readable formal specification:

- **`primitives.yaml`** — Shape vocabulary: tiles `T(M,N)`, oscillators `O(μ,ω₀)`, illusions `Ψ(stim,percept)`, and 15 other primitives with parameters, equations, and examples
- **`rules.yaml`** — 15 within-layer production rules, 5 cross-layer isomorphisms, causal uncertainty rules, and computational psychiatry mappings
- **`isomorphisms.yaml`** — Deep analogies (dispatch↔tonotopy, fusion↔texture, split-K↔cocktail party) with code and equation references
- **`visualize.py`** — Matplotlib renderer for grammar diagrams (requires `matplotlib`, `pyyaml`)

```bash
# Generate the grammar overview diagram
cd grammar && python visualize.py
```

## Demos

### Nonstationary Audio Analysis

Spectral mixture Gaussian process regression on a synthetic duet — a chirp (200→600 Hz) mixed with an amplitude-modulated 440 Hz tone. The GP captures the signal's frequency content probabilistically, with uncertainty, rather than through a fixed-resolution spectrogram.

See [`notebooks/nonstationary_audio.py`](notebooks/nonstationary_audio.py) — inspired by [Wilkinson et al. (2019)](https://arxiv.org/abs/1901.11436).

### EP via Kalman Smoothing

Implementation of Algorithm 1 from Wilkinson et al. — Expectation Propagation inference in a state-space GP model via Kalman filtering (forward pass) and RTS smoothing (backward pass). The smoother consistently outperforms the filter (RMSE ~0.16 vs ~0.21) because it incorporates future observations.

See [`notebooks/ep_kalman_viz.py`](notebooks/ep_kalman_viz.py).

## Sources

See [`sources/BIBLIOGRAPHY.md`](sources/BIBLIOGRAPHY.md) for an annotated bibliography of the collected research papers, organized by theme:

- **Physics of Sound & Signal Analysis** — probabilistic audio inference, physical sound synthesis
- **Mathematical Foundations** — algebraic QFT, superselection structure
- **Cognition of Hearing & Neuroscience** — auditory processing, neural correlates

## License

[MIT](LICENSE) — Copyright (c) 2020 iiG
