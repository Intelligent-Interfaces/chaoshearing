# Chaos Hearing

_[Erick Oduniyi](https://eoduniyi.github.io/erick.light/)_

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
├── sources/                  # Research papers (PDFs) and annotated bibliography
│   ├── BIBLIOGRAPHY.md       # Annotated guide to collected papers
│   └── *.pdf                 # Source papers
├── projects/
│   ├── 01-listening-and-duets.md    # Chaotic dynamics & auditory scene analysis
│   └── 02-spectral-cognition.md     # Sound physics, musical imagery & perception
├── notebooks/
│   └── nonstationary_audio.py       # GP-based nonstationary audio analysis demo
├── docs/                     # GitHub Pages site
│   ├── index.md
│   └── projects/
└── _config.yml
```

## Projects

### [01 — On Listening and Duets](projects/01-listening-and-duets.md)

_Hearing Performance, Chaotic Dynamics, and Computational Auditory Scene Analysis_

The original thread: modeling the cochlea and auditory perception through the lens of nonlinear dynamics (Hopf bifurcations, limit cycles) and connecting that to how we parse complex sound scenes — duets, ensembles, overlapping conversations.

### [02 — Spectral Cognition](projects/02-spectral-cognition.md)

_Sound Physics, Musical Imagery, and the Perception of Pitch_

The newer thread: bridging the physics of sound (synthesis, physical modeling, probabilistic spectral analysis) with cognitive phenomena — perfect pitch, involuntary musical imagery, auditory hallucinations, and what these tell us about how the brain constructs sound.

## Demos

### Nonstationary Audio Analysis

A Python implementation exploring spectral mixture Gaussian processes for audio signal analysis — inspired by [Wilkinson et al. (2019)](https://arxiv.org/abs/1901.11436). See [`notebooks/nonstationary_audio.py`](notebooks/nonstationary_audio.py).

## Sources

See [`sources/BIBLIOGRAPHY.md`](sources/BIBLIOGRAPHY.md) for an annotated bibliography of the collected research papers.

## License

[MIT](LICENSE) — Copyright (c) 2020 iiG
