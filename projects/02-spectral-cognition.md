---
layout: default
title: "02 — Spectral Cognition"
---

# 02 — Spectral Cognition

_Sound Physics, Musical Imagery, and the Perception of Pitch_

[← Home]({{ site.baseurl }}/)

**Erick Oduniyi**

---

## Motivation

You're walking down the street and a song starts playing in your head. Not from headphones — from _inside_. The melody is vivid, the timbre recognizable, the pitch precise. Where is this sound? What is your brain doing?

This project explores the space between the physics of sound and the cognition of hearing — particularly the phenomena that reveal how deeply the brain _constructs_ auditory experience rather than passively receiving it.

## Threads

### Perfect Pitch (Absolute Pitch)

About 1 in 10,000 people can identify or produce a musical note without a reference tone. This isn't just good ears — it's a cognitive labeling system that maps continuous frequency space onto discrete categories. Questions:

- Is absolute pitch a perceptual phenomenon (enhanced frequency resolution) or a memory phenomenon (stable long-term pitch templates)?
- What does it tell us about the relationship between continuous physical signals and discrete mental representations?
- How does it relate to language — tonal language speakers show higher rates of absolute pitch?

### Involuntary Musical Imagery (Earworms)

Nearly everyone experiences involuntary musical imagery — songs "stuck" in your head. This is the auditory system running in generative mode without external input. It reveals:

- The brain maintains detailed spectral and temporal models of sound
- These models can be activated endogenously (without stimulus)
- The line between "hearing" and "imagining" is thinner than we think

### Auditory Hallucinations

In psychosis, the generative capacity of the auditory system goes further — producing voices, music, or sounds that are experienced as fully external and real. This is not a failure of hearing but an _excess_ of the same constructive process that lets us parse cocktail parties and imagine melodies.

The continuum: **auditory scene analysis → musical imagery → auditory hallucination** — all manifestations of the brain's predictive model of sound, operating at different levels of constraint.

### The Jazz of Physics (Improvisation as Spectral Search)

Stephon Alexander's _The Jazz of Physics_ (2016) argues that jazz improvisation and theoretical physics share deep structural parallels — both are acts of pattern recognition, constraint satisfaction, and creative exploration within formal systems. This is not loose metaphor. Alexander draws a direct line from John Coltrane's harmonic substitution patterns ("Coltrane Changes" — the symmetric divisions of the octave that generate _Giant Steps_) to symmetry principles in cosmology and quantum gravity.

The connection to spectral cognition is concrete:

- **Coltrane's circle**: The Coltrane Changes divide the octave into major thirds (C → E → A♭ → C), forming an augmented triad — a geometric symmetry on the circle of fifths. This is a _group-theoretic_ structure: the cyclic group Z₃ acting on the 12-tone chromatic space Z₁₂. Coltrane was reportedly influenced by Einstein and by the geometry of pitch space.
- **Acoustic oscillations in the early universe**: The cosmic microwave background contains acoustic oscillations — literal sound waves in the early universe's plasma. The CMB power spectrum is the frequency spectrum of the universe's first sound. Alexander connects this to how we analyze musical sound — the same Fourier decomposition, the same spectral peaks, the same harmonic structure.
- **Improvisation as real-time search**: A jazz solo is a real-time search through a high-dimensional space of possible note sequences, constrained by harmonic structure (the chord changes), rhythmic feel (the groove), and the responses of other musicians (interaction). This mirrors how physicists explore solution spaces — you have constraints (symmetries, conservation laws), you have a landscape of possibilities, and the creative act is navigating that landscape in real time.

The improvisation framing connects directly to the computational models in this project: the GP spectral inference framework is a formal version of what a jazz musician does — maintain a probabilistic model of the harmonic context (the prior), update it continuously as new notes arrive (the likelihood), and use it to predict, select, and generate the next phrase (the posterior). The spectral mixture kernel _is_ the chord chart.

### Probabilistic Spectral Analysis

From the signal processing side, [Wilkinson et al. (2019)](https://arxiv.org/abs/1901.11436) show that audio analysis can be formulated as Gaussian process inference with spectral mixture kernels. This is interesting not just as engineering but as a _model of perception_:

- The GP prior encodes expectations about spectral structure (harmonicity, smoothness)
- Inference updates these expectations given observed data
- Nonstationarity is handled naturally — the model adapts as the signal changes

This mirrors what the auditory system does: maintain a probabilistic model of the sound scene, update it continuously, and use it to predict, separate, and interpret incoming signals.

## The Bridge

The physics of sound production (nonlinear dynamics, contact mechanics, wave propagation) determines what signals arrive at the ear. The cognition of hearing (scene analysis, pitch perception, imagery) determines what we _experience_. The bridge between them is **spectral cognition** — the computational and neural processes that transform physical vibration into perceptual meaning.

```
Production → Propagation → Transduction → Representation → Perception → Imagery
   ↑                                                                        ↓
   └──────────── Interface Design (closing the loop) ──────────────────────┘
```

Building interfaces for sound means understanding this entire chain — and finding the right points to give people control.

## Open Questions

- Can probabilistic audio models (GPs, state-space models) serve as computational analogues of auditory predictive processing?
- What distinguishes the neural dynamics of veridical hearing, musical imagery, and auditory hallucination?
- How should audio interfaces represent uncertainty and ambiguity in sound scenes?
- What can physical sound synthesis (with perceptual constraints) teach us about the features the auditory system actually uses?
- Can the group-theoretic structure of Coltrane's harmonic substitutions be formalized as production rules in a shape grammar over pitch space?
- What is the relationship between improvisation (real-time search under harmonic constraints) and auditory scene analysis (real-time inference under grouping constraints)?

## Related Sources

- [End-to-end probabilistic inference for nonstationary audio](../sources/BIBLIOGRAPHY.md#end-to-end-probabilistic-inference-for-nonstationary-audio-analysis) — `1901.11436v5.pdf`
- [Object-based sound synthesis](../sources/BIBLIOGRAPHY.md#object-based-synthesis-of-scraping-and-rolling-sounds-based-on-non-linear-physical-constraints) — `2112.08984v1.pdf`
- [Superselection structure in QFT](../sources/BIBLIOGRAPHY.md#superselection-structure-of-massive-quantum-field-theories-in-11-dimensions) — `9705019v1.pdf`
- Alexander, S. (2016). _The Jazz of Physics: The Secret Link Between Music and the Structure of the Universe_. Basic Books.
