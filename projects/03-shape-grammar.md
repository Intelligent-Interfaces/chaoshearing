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

| Concept       | Backend (TileGym)     | Interface (Chaos Hearing)   | Frontend (McDermott Demos)  |
| :------------ | :-------------------- | :-------------------------- | :-------------------------- |
| Primitive     | Tile `T(M,N)`         | Oscillator `O(μ,ω₀)`        | Illusion `Ψ(stim,percept)`  |
| Decomposition | Tile blocking         | Cochlear frequency analysis | Auditory scene analysis     |
| Dispatch      | Backend selector      | Frequency-place mapping     | Perceptual frame selection  |
| Fusion        | Kernel fusion         | Spectral mixture kernel     | Texture perception          |
| Reduction     | Split-K merge         | Scene integration           | Source segregation          |
| Attention     | `A(Q,K,V)`            | Bayesian prediction         | Grouping cues               |
| Normalization | RMS/LayerNorm         | Cochlear compression        | Perceptual continuity       |
| Improvisation | Autoregressive decode | Jazz solo over changes      | Generative listening (BASS) |
| Failure mode  | Wrong dispatch        | Hallucination               | Illusion                    |
| Uncertainty   | Backend fallback      | Posterior variance          | Causal ambiguity            |

---

## Worked Examples

Eight examples trace each isomorphism through concrete code, equations, and perceptual scenarios.

### 1. Fusion — SwiGLU to Spectral Mixture to Texture

TileGym's `PartiallyFusedSwiGLUMLP` fuses five kernel launches into three by concatenating gate and up projection weights and eliminating intermediate tensors. The spectral mixture kernel fuses multiple frequency components into a single covariance function — no intermediate spectrogram. Texture perception fuses moment-by-moment spectral frames into a single statistical summary — listeners can't distinguish original rain from synthetic rain with matched statistics.

**The isomorphism**: Fusion eliminates intermediate representations at every level.

### 2. Split-K Reduction — Parallel Tiles to the Cocktail Party

Flash decode splits a long KV cache across thread blocks, each computing partial attention and log-sum-exp independently, then `splitk_reduce` merges them. The cochlea splits an acoustic mixture across ~3,500 frequency-tuned oscillators, then the auditory system reduces them into coherent streams using grouping cues (harmonicity, onset, proximity). The EP Kalman smoother in [`notebooks/ep_kalman_viz.py`]({{ site.baseurl }}/notebooks/ep_kalman_viz.py) implements a temporal version: forward filter + backward smoother = split-and-merge in time.

**The isomorphism**: Split into parallel channels → process independently → merge using structural cues.

### 3. Dispatch — Backend Selection to Tonotopy to Perceptual Framing

TileGym's `@dispatch` decorator routes operations to cutile (primary) or pytorch (fallback). The cochlea's tonotopic map routes each frequency to its tuned oscillator — the Hopf bifurcation model amplifies matched frequencies ~1000× and attenuates mismatches. The McDermott Lab schema learning demos show listeners routing sounds to learned perceptual templates.

**The isomorphism**: Route input to specialized processor based on properties, with fallback for unmatched cases.

### 4. Attention — QKV to Bayesian Prediction to Auditory Grouping

TileGym's attention sink mechanism maintains learned "sink tokens" that absorb leftover attention mass — a prior that says "if nothing is relevant, attend to me." The auditory system's predictive model maintains a prior over expected sounds — when data is ambiguous, the prior shapes the percept. The grouping demos show a phase transition: slow tone sequences → one stream (broad attention), fast sequences → two streams (focused attention).

**The isomorphism**: Selectively weight information based on relevance, with a prior/sink absorbing irrelevant context.

### 5. Normalization — RMSNorm to Cochlear Compression to Illusory Continuity

RMSNorm divides by root-mean-square, preserving relative structure regardless of input magnitude. The cochlea's compressive nonlinearity (response ∝ input^(1/3) near Hopf criticality) compresses 120 dB of dynamic range into ~40 dB of neural firing. Spectral completion demos show sounds perceived as continuing behind loud noise bursts — the auditory system "normalizes out" the interruption.

**The isomorphism**: Stabilize representations against magnitude variation, preserving relative structure.

### 6. Mixture of Experts — MoE Routing to Selective Listening

MoE layers route each token to its top-K experts (e.g., 6 out of 64), with `moe_align_block_size` sorting tokens by expert for efficient tiling. Selective auditory attention routes processing resources to the attended stream, suppressing others. The attentive tracking demos measure how well listeners follow one melody among four simultaneous melodies.

**The isomorphism**: A gating mechanism selects a sparse subset of processors, concentrating resources on relevant inputs.

### 7. Causal Uncertainty — Dispatch Ambiguity to Posterior Variance to Bistability

TileGym's fallback chain (cutile → pytorch → error) is graceful degradation under implementation uncertainty. The GP spectral analysis represents causal uncertainty as posterior variance — high where the signal is ambiguous, low where it matches the prior. Inharmonic speech demos produce perceptual bistability — the auditory Necker cube — when harmonicity cues are degraded.

**The isomorphism**: When the input-to-processor mapping is ambiguous, manage uncertainty through degradation, probabilistic representation, or alternating interpretations.

### 8. Improvisation — Autoregressive Decoding to Jazz Solos to Generative Listening

_After Stephon Alexander, "The Jazz of Physics" (2016)_

LLM autoregressive decoding generates tokens one at a time, each conditioned on the KV cache (context) and shaped by model weights (prior), with sampling temperature controlling exploration vs. exploitation. A jazz musician generates notes one at a time, conditioned on the chord changes and shaped by harmonic knowledge, with the "inside/outside" continuum as the temperature parameter. The BASS generative listening model infers scene decompositions conditioned on the acoustic mixture and shaped by source priors.

#### Coltrane Changes as Production Rule

John Coltrane's harmonic substitutions in _Giant Steps_ (1960) divide the octave into major thirds — the cyclic group Z₃ acting on the 12-tone chromatic space Z₁₂:

```
Standard ii-V-I in C:    Dm7 → G7 → Cmaj7

Coltrane substitution:   Dm7 → G7 → Cmaj7
                              ↘ E7 → Amaj7
                                   ↘ B♭7 → E♭maj7

Resolution targets: C → A♭ → E → C  (major thirds = Z₃ orbit)
```

This is a production rule on harmonic space: `I → I · T₄ · T₄ · T₄`

#### The Spectral Mixture Kernel as Chord Chart

The GP spectral mixture kernel formalizes the harmonic prior:

```python
# A jazz chord as a spectral mixture kernel:
# Cmaj7 = C(262Hz) + E(330Hz) + G(392Hz) + B(494Hz)
weights   = [1.0, 0.8, 0.9, 0.7]           # voicing weights
means     = [262.0, 330.0, 392.0, 494.0]    # chord tone frequencies
variances = [50.0, 50.0, 50.0, 50.0]        # tolerance (inside vs outside)
```

Tight variances = playing inside the changes (bebop). Wide variances = venturing into chromatic territory (free jazz). This is the same temperature parameter as in LLM sampling.

#### Cosmic Inharmonicity

Alexander's most striking connection: the CMB acoustic oscillation peaks are at ratios ~1 : 2.45 : 3.68 — not quite harmonic. The baryon-to-photon ratio shifts the overtones away from integer multiples, just as the inharmonic speech demos shift partials away from harmonicity. The universe's first sound is a cosmic instance of the inharmonicity manipulation that probes auditory grouping.

| Concept     | Backend (LLM Decode) | Interface (Jazz Solo)   | Frontend (BASS Listening) |
| :---------- | :------------------- | :---------------------- | :------------------------ |
| Context     | KV cache             | Chord changes + history | Acoustic mixture          |
| Prior       | Model weights        | Harmonic knowledge      | Sound source priors       |
| Generation  | Sample from logits   | Play next note          | Infer scene decomposition |
| Constraint  | Causal mask          | Chord tones, rhythm     | Physical plausibility     |
| Temperature | Sampling temperature | Inside/outside playing  | Posterior sharpness       |
| Compression | MLA latent KV        | Internalized changes    | Statistical summary       |

**The isomorphism**: Real-time sequential generation under constraints, where each step is conditioned on accumulated context and shaped by a learned prior, with a temperature parameter controlling exploration-exploitation.

---

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

The grammar suggests that **perceptual disorders are production rule failures** — the same rules that enable normal hearing can malfunction in specific, characterizable ways.

## Grammar Specification

The formal grammar is defined in machine-readable YAML:

- [`grammar/primitives.yaml`]({{ site.baseurl }}/grammar/primitives.yaml) — Shape vocabulary
- [`grammar/rules.yaml`]({{ site.baseurl }}/grammar/rules.yaml) — Production rules (including improvisation rules IMP1–IMP5)
- [`grammar/isomorphisms.yaml`]({{ site.baseurl }}/grammar/isomorphisms.yaml) — Cross-layer mappings
- [`grammar/visualize.py`]({{ site.baseurl }}/grammar/visualize.py) — Diagram renderer

## Related Sources

- [End-to-end probabilistic inference for nonstationary audio](../sources/BIBLIOGRAPHY.md#end-to-end-probabilistic-inference-for-nonstationary-audio-analysis) — `1901.11436v5.pdf`
- [Object-based sound synthesis](../sources/BIBLIOGRAPHY.md#object-based-synthesis-of-scraping-and-rolling-sounds-based-on-non-linear-physical-constraints) — `2112.08984v1.pdf`
- Cusimano, Hewitt, & McDermott (2024). "Listening with generative models." _Cognition_, 253.
- Alexander, S. (2016). _The Jazz of Physics: The Secret Link Between Music and the Structure of the Universe_. Basic Books.
- Stiny, G. (1980). "Introduction to shape and shape grammars." _Environment and Planning B_.
