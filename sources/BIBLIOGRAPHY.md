# Annotated Bibliography

A guide to the research papers collected in `sources/`, organized by theme.

---

## Physics of Sound & Signal Analysis

### End-to-End Probabilistic Inference for Nonstationary Audio Analysis

**Wilkinson et al., 2019** — `1901.11436v5.pdf`
[arXiv:1901.11436](https://arxiv.org/abs/1901.11436)

Proposes a unified Gaussian process framework that jointly performs time-frequency analysis and nonnegative matrix factorization for audio signals. Uses spectral mixture kernels with nonstationary priors and a state-space formulation for scalable inference (linear in time steps). Directly relevant to building principled, probabilistic audio analysis pipelines that don't rely on ad-hoc spectrogram stages.

**Why it matters for Chaos Hearing:** This paper shows how to replace the traditional "compute spectrogram → extract features" pipeline with a single probabilistic model. It connects to our interest in understanding the _structure_ of sound — not just its surface representation — and opens the door to audio analysis methods that can adapt to nonstationary, real-world signals (music, speech, environmental sound).

---

### Object-Based Synthesis of Scraping and Rolling Sounds Based on Non-Linear Physical Constraints

**Agarwal et al., 2021** — `2112.08984v1.pdf`
[arXiv:2112.08984](https://arxiv.org/abs/2112.08984)

Presents a source-filter model for synthesizing sustained contact sounds (scraping, rolling) with physically and perceptually meaningful parameters. Key innovations include nonlinear contact force constraints, naturalistic normal force variation, and location-dependent impulse response morphing. Perceptual experiments validate realism.

**Why it matters for Chaos Hearing:** This is physical sound synthesis done right — grounded in mechanics but validated by human perception. It bridges the physics thread (how sound is _produced_) with the cognition thread (how sound is _heard_). The nonlinear constraints echo the chaotic dynamics we study in auditory models.

---

## Mathematical Foundations

### Superselection Structure of Massive Quantum Field Theories in 1+1 Dimensions

**Müger, 1997** — `9705019v1.pdf`
[arXiv:hep-th/9705019](https://arxiv.org/abs/hep-th/9705019)

Analyzes the superselection structure of massive QFTs in 1+1 dimensions using algebraic quantum field theory (Haag duality, split property). Shows that certain representation-theoretic structures are vacuous for massive theories but rich for massless/conformal theories.

**Why it matters for Chaos Hearing:** The algebraic structure of wave phenomena — particularly the distinction between massive and massless theories — provides deep mathematical analogies for understanding sound propagation, resonance, and the structure of auditory representations. The conformal invariance of massless theories connects to scale-invariant properties of pitch perception and harmonic structure.

---

## Cognition of Hearing & Neuroscience

### bioRxiv 2026.03.12.711349v1

`2026.03.12.711349v1.full (1).pdf`

_(Auditory neuroscience / cognitive hearing — paper details to be annotated upon review.)_

**Provisional relevance:** Likely connects to the cognitive thread — auditory processing, neural correlates of sound perception, or related neuroscience of hearing.

---

### SSRN 5827765

`ssrn-5827765.pdf`

_(Social science / cognitive science of hearing — paper details to be annotated upon review.)_

**Provisional relevance:** Likely connects to the broader cognitive and social dimensions of hearing — perception, musical experience, or auditory phenomena in clinical populations.

---

## Themes Across the Collection

The papers cluster around a central question: **How does sound go from physics to perception?**

```
Physical Production          Signal Structure          Neural Processing
(mechanics, nonlinear    →   (spectral analysis,   →  (auditory scene analysis,
 dynamics, synthesis)         GP models, TF reps)      imagery, hallucination)
```

The Chaos Hearing project lives in the arrows between these stages — understanding the transformations, building computational models of them, and designing interfaces that let people interact with sound at each level.
