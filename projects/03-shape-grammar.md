---
layout: default
title: "03 — Shape Grammar"
---

# 03 — Shape Grammar

_A Layered Grammar Connecting Tile-Based GPU Computing, Auditory Science, and Perceptual Demos_

[← Home]({{ site.baseurl }}/)

**Erick Oduniyi**

---

## Motivation

Shape grammars are production systems: start with an initial shape, apply transformation rules, generate a language of forms. This project defines a shape grammar that operates across three layers of a single perceptual computing stack — from GPU tiles to cochlear oscillators to auditory illusions.

The claim: **the same structural patterns — tiling, dispatch, fusion, decomposition, and reduction — recur at every level**, from silicon to synapse to subjective experience.

## The Three Layers

| Layer         | Project                                                         | Role                                                                                        |
| :------------ | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Backend**   | [TileGym](https://github.com/NVIDIA/TileGym)                    | Computational substrate — how tiles of data are dispatched, fused, and reduced on hardware  |
| **Interface** | [Chaos Hearing]({{ site.baseurl }}/)                            | Scientific model — how sound is produced, transduced, represented, and perceived            |
| **Frontend**  | [McDermott Lab Demos](https://mcdermottlab.mit.edu/sounds.html) | Perceptual surface — interactive demonstrations that expose the grammar to human experience |

## The Meta-Rule

Every layer instantiates the same five-stage pipeline:

```
INPUT (complex, entangled)
  → DECOMPOSE into tiles / oscillators / streams
  → DISPATCH to specialized processors / channels / frames
  → TRANSFORM via fused kernels / spectral inference / prediction
  → REDUCE via split-K / scene integration / perceptual binding
  → OUTPUT (structured, interpreted)
```

## Cross-Layer Isomorphisms

| Concept       | Backend (TileGym) | Interface (Chaos Hearing)   | Frontend (McDermott Demos) |
| :------------ | :---------------- | :-------------------------- | :------------------------- |
| Primitive     | Tile `T(M,N)`     | Oscillator `O(μ,ω₀)`        | Illusion `Ψ(stim,percept)` |
| Decomposition | Tile blocking     | Cochlear frequency analysis | Auditory scene analysis    |
| Dispatch      | Backend selector  | Frequency-place mapping     | Perceptual frame selection |
| Fusion        | Kernel fusion     | Spectral mixture kernel     | Texture perception         |
| Reduction     | Split-K merge     | Scene integration           | Source segregation         |
| Attention     | `A(Q,K,V)`        | Bayesian prediction         | Grouping cues              |
| Failure mode  | Wrong dispatch    | Hallucination               | Illusion                   |

## Causal Uncertainty in Sound Objects

When you hear a sound, your brain infers its cause. This inference is often ambiguous — the same acoustic signal could have been produced by different physical events. The grammar formalizes this:

- **Backend**: Dispatch ambiguity — the same operation routes to different backends producing equivalent results through different paths
- **Interface**: The GP spectral inference framework represents causal uncertainty as posterior variance over possible decompositions
- **Frontend**: Demos on inharmonic speech and illusory texture manipulate causal uncertainty directly

## Computational Psychiatry

The grammar extends into computational psychiatry through predictive processing:

| Condition         | Grammar Rule                     | Description                     |
| :---------------- | :------------------------------- | :------------------------------ |
| Normal perception | `P(prior) + data → P(posterior)` | Balanced Bayesian update        |
| Musical imagery   | `P(prior) + ∅ → I(vivid)`        | Prior-driven generation         |
| Hallucination     | `P(prior) ≫ data → Ψ(false)`     | Prior dominance                 |
| Tinnitus          | `O(locked) → persistent`         | Oscillator stuck in limit cycle |

## Grammar Specification

The formal grammar is defined in machine-readable YAML:

- [`grammar/primitives.yaml`]({{ site.baseurl }}/grammar/primitives.yaml) — Shape vocabulary
- [`grammar/rules.yaml`]({{ site.baseurl }}/grammar/rules.yaml) — Production rules
- [`grammar/isomorphisms.yaml`]({{ site.baseurl }}/grammar/isomorphisms.yaml) — Cross-layer mappings
- [`grammar/visualize.py`]({{ site.baseurl }}/grammar/visualize.py) — Diagram renderer

## Related Sources

- [End-to-end probabilistic inference for nonstationary audio](../sources/BIBLIOGRAPHY.md#end-to-end-probabilistic-inference-for-nonstationary-audio-analysis) — `1901.11436v5.pdf`
- [Object-based sound synthesis](../sources/BIBLIOGRAPHY.md#object-based-synthesis-of-scraping-and-rolling-sounds-based-on-non-linear-physical-constraints) — `2112.08984v1.pdf`
- Cusimano, Hewitt, & McDermott (2024). "Listening with generative models." _Cognition_, 253.
- Stiny, G. (1980). "Introduction to shape and shape grammars." _Environment and Planning B_.

---

See the full grammar document: [`SHAPE_GRAMMAR.md`](../../SHAPE_GRAMMAR.md)
