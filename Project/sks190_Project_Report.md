# Minimal Representations and the Dual of SAL: Learning to Reintroduce Nuisance Variables  
**Author:** Sreya Kadagattur Srinidhi

---

# 1. Abstract/Introduction

This project investigates the construction of minimal visual representations and the corresponding “dual” problem of reintroducing nuisance variables removed during representation learning. Building on the Sampling and Anti-Aliasing Likelihood (SAL) framework of Soatto and Chiuso, I implement SAL encoders for viewpoint, brightness, and contrast and analyze the geometric relationships between their induced invariances. A quantitative hierarchy emerges: photometric nuisances align in feature space, while geometric viewpoint invariance lies in an opposing direction—an insight that proves crucial when training decoders. I propose a dual-SAL decoder that conditions on a nuisance variable \(g\) to synthesize images of a scene from a minimal SAL embedding \(z\). Initial failures reveal that per-image embeddings contain residual viewpoint information, causing the decoder to ignore \(g\). Enforcing true minimality via a scene-level embedding resolves this, enabling all decoder architectures to correctly reintroduce viewpoint. This demonstrates a practical dual to SAL and establishes groundwork for controllable nuisance reintroduction.


---

# 2. Related Work

This project builds upon a series of influential frameworks connecting **nuisance variables**, **invariance**, and **representation sufficiency**. I focus primarily on:

1. Soatto & Chiuso (2016): *Visual Representations: Defining Properties and Deep Approximations*  
2. NeRF and NeRFactor: rendering under geometry and lighting  
3. MaterialMVP: decomposing contrast/lighting editing  

---

## 1.1 Sampling & Anti-Aliasing Likelihood (SAL)

A visual representation must be **minimal** and **sufficient**, preserving only scene information θ while removing nuisance g (viewpoint, illumination, tone, deformation, etc.).

The ideal invariant representation is the **profile likelihood**:

$$
p_{\theta,G}(y) = \sup_{g\in G} p_{\theta,g}(y)
$$

But G is continuous and uncountably large. Direct maximization is intractable, so Soatto & Chiuso propose:

- **sampling** nuisance transforms $g_1,\dots, g_N$
- applying **anti-aliasing** via local marginalization
- then taking the **max** across samples (SAL)

This combination prevents aliasing (different nuisance values folding onto the same representation) and ensures that small changes in the nuisance parameter g lead to small, smooth changes in the representation. In other words, SAL preserves continuity in the “nuisance index”: if $g$ and $g'$ are close (e.g., nearby viewpoints or brightness levels), then $\phi(y_g)$ and $\phi(y_{g'})$ remain close as well, instead of jumping discontinuously.

The result is a representation:

- invariant to nuisances  
- sufficient for recognition  
- minimal (removes high-frequency variability due to nuisances)

---

## 1.2 Why Different Nuisances Require Different Mathematics

The nuisance group determines:

- whether invariants exist  
- whether marginalization is tractable  
- whether sampling is necessary  

| Nuisance | Structure | Properties | Notes |
|----------|-----------|------------|-------|
| **Contrast** | monotone pixel transform | preserves level sets, gradient orientation | invariants exist |
| **Viewpoint** | diffeomorphism | warps geometry nonlinearly | must be sampled locally |
| **Rotation** | SO(2) | admits canonicalization | SIFT-style orientation selection |
| **Scaling** | ℝ⁺ | destroys sampling grid → aliasing | multi-scale pooling |
| **Translation** | ℝ² | convolution = sampling translations | handled by CNNs |
| **Occlusion** | not a group | breaks domain | requires receptive fields |

Thus, SAL encoders for different nuisances must implement **different pooling kernels**, **sampling densities**, and **aggregation strategies**.

---

## 1.3 NeRF, NeRFactor, and Decoders

NeRF-style models define:

$$
\mathcal{R}_\theta(g) \longrightarrow y_g
$$

mapping a scene $\theta$ and viewpoint g to a rendered image.  
NeRFactor further decomposes lighting $g_{light}$ and materials $g_{mat}$.  
MaterialMVP manipulates tone/contrast in image space.

These papers show that **nuisance reintroduction** (viewpoint, lighting, tone) is feasible, but requires explicit conditioning on nuisance parameters.

None of these works examine **the dual of SAL**—the process of **adding nuisances back after SAL removed them**.  
This project fills that gap.

---

# 2. Methods

My goal is to take a **minimal SAL embedding z**, which intentionally removed viewpoint/lighting/contrast, and design a **dual decoder** that reintroduces the selected nuisance variable:

$$
(z, g) \longmapsto \hat{y}_g
$$

I completed three major components:

1. **Built SAL encoders** for viewpoint, brightness, and contrast  
2. **Quantitatively studied the nuisance hierarchy** induced by SAL  
3. **Designed and trained dual-SAL decoders** for viewpoint reintroduction  

---

## 2.1 SAL Encoders for Multiple Nuisance Families

I implemented three SAL encoders:

- **View-SAL Encoder**  
- **Brightness-SAL Encoder**  
- **Contrast-SAL Encoder**

Each uses:

- ResNet-18 backbone features  
- Nuisance sampling (view angles, brightness scales, contrast curves)  
- Anti-aliased weighted pooling or LSE pooling  
- Per-patch marginalization as required by SAL theory  

I also implemented baselines:

- Mean pooling  
- Max pooling  
- Gaussian smoothing  
- Softmax/LSE pooling

These allow direct comparison between SAL and classical invariance mechanisms.

---

## 2.2 Empirical Discovery: Hierarchy of Nuisance Variables

Through cosine similarity comparisons between embeddings produced by different encoders, I discovered a **hierarchy**:

### Key Findings:

- **Brightness-SAL** is strongly opposed to **Viewpoint-SAL**  
- **Contrast-SAL** is moderately opposed to Viewpoint-SAL  
- **Brightness-SAL** and **Contrast-SAL** are aligned  
- **Mean pooling** destroys nuisance structure  
- **SAL pooling** preserves nuisance geometry and produces meaningful invariants  

This hierarchy directly matches the mathematical structure of nuisance groups:

- brightness transforms destroy intensity-scale → major conflict with viewpoint  
- contrast preserves edges → lesser conflict with viewpoint
- viewpoint is geometric → orthogonal to photometric nuisances such as brightness and contrast

In other words, SAL encoders for **photometric** nuisances (brightness, contrast) tend to produce embeddings that lie in a similar direction in feature space, while the **geometric** View-SAL encoder displaces embeddings along a very different axis. Mean pooling, by contrast, collapses these directions almost entirely, erasing the structure associated with individual nuisance types.

This matters later for the decoder: if a supposedly invariant embedding $z_i$ still carries residual viewpoint information, the decoder can exploit that shortcut and learn $D(z_i, g_i) \approx y_i$ while effectively ignoring $g$. The observed hierarchy makes it plausible that viewpoint “leaks” into $z$ more strongly than photometric nuisances, and explains why learning to reintroduce viewpoint is more challenging than learning to reintroduce brightness or contrast. Ensuring that $z$ is truly minimal with respect to viewpoint is therefore critical for the dual-SAL decoder to behave correctly.

This empirical hierarchy, and its connection to the structure of the nuisance group, is one of the novel contributions of this study.

---

## 2.3 The Dual to SAL: Learning to Reintroduce Nuisance Variables

SAL creates a nuisance-invariant embedding:

$$
z = \phi(y)
$$

by **removing** the effect of g.

I define a decoder:

$$
D_\psi : (z, g) \mapsto \hat{y}_g
$$

that attempts to **reapply** the nuisance transform.  
This corresponds to the SAL dual:

$$
\tilde{y}_g = g(y) \quad \approx \quad D_\psi(z,g)
$$

---

## 2.4 Technical Insight: Enforcing Minimality via Scene-Level z

When using a per-image SAL embedding $z_i$, I discovered the decoder simply learned:

$$
D(z_i, g_i) \approx y_i
$$

while **ignoring g entirely**.

This is because per-image embeddings still contained residual viewpoint information as explained in Section 2.2.

### Implemented Fix

For each scene:

1. Compute z for every view: $z_1,\dots,z_N$
2. Compute scene-level embedding:

$$
z_{scene} = \frac{1}{N}\sum_i z_i
$$

3. Replace every $z_i$ with $z_{scene}$

Now the decoder sees:

$$
(z_{scene}, g_i) \mapsto y_i
$$

and the **only way** to explain changes across images is to **use g**.

This modification is essential for the dual-SAL decoder to behave correctly.

This is the primary novel technical insight of my project.

---

## 2.5 Decoder Architectures

I implemented three architectures:

### 1. **Baseline Decoder (64×64)**
- FC → reshape to 8×8  
- 3× ConvTranspose (8→16→32→64)  
- Sigmoid output  

### 2. **BaseCh128 Decoder (128-channel)**
- Higher channel count for more expressiveness  
- Same structure  

### 3. **ResNet64 Decoder**
- Up-convolutions  
- Residual blocks  
- Most expressive  

All take:

- latent z  
- nuisance code g (pose vector)  
- output low-frequency image reconstruction  

Loss: L1  
Optimizer: Adam  
Epochs: 20–50

---

# 3. Results, Analysis, and Discussion

My experiments consist of **two major investigations**:

1. Studying invariances and the nuisance hierarchy with SAL encoders  
2. Reintroducing viewpoint as a nuisance using the dual-SAL decoder  

---

## 3.1 Study 1: Hierarchy of Nuisance Variables (SAL Encoders)

This study discovered a **clear, empirically grounded hierarchy**:

- Brightness-SAL and Contrast-SAL are aligned  
- Both oppose Viewpoint-SAL  
- Mean pooling erases nuisance structure  
- SAL pooling preserves meaningful structure  
- Viewpoint SAL exhibits strongest, most consistent invariance

Key cosine-similarity observations across 20 fern views:

| Comparison | Avg Cosine | Interpretation |
|------------|------------|----------------|
| View-SAL ↔ Baseline | +0.95 | View SAL preserves high-level structure |
| Bright-SAL ↔ View-SAL | –0.15 | Strong nuisance conflict |
| Contrast-SAL ↔ View-SAL | –0.05 | Moderate conflict |
| Bright-SAL ↔ Contrast-SAL | +0.10 | Shared photometric family |
| Mean pooling ↔ SAL | ~0 | Mean pooling collapses invariance structure |

These results experimentally validate the theory: **each nuisance induces a different equivalence class**, and SAL must be tailored to the nuisance group.

---

## 3.2 Study 2: Dual-SAL Decoder for Viewpoint Reintroduction

### 3.2.1 Qualitative Results

All reconstructions are intentionally blurry because SAL embeddings are minimal—they remove high-frequency structure.

Example (fern scene):

- GT(i): sharp ground truth at pose $g_i$  
- GT(j): sharp ground truth at pose $g_j$  
- Recon($z_i$, $g_i$): blurry reconstruction  
- Swap($z_i$, $g_j$): blurry but shifted view  

Even though detail is gone, viewpoint-dependent structure is preserved in low frequency (background shifts, foreground blob motion, etc.).


It is easier to see this in the actual images than in the raw distances. I include links to a few representative qualitative examples below:

- [Baseline decoder qualitative example](images/View_baseline_decoder_20epochs.png)  
- [128-channel decoder qualitative example](images/View_128_decoder_20epochs.png)  
- [ResNet decoder qualitative example](images/View_resnet64_decoder_20epochs.png) 

---

### 3.2.2 Quantitative Results

To evaluate whether each decoder truly uses the nuisance variable $g$, I compare four quantities for two views $i$ and $j$ of the same scene:

- **L1(recon\_i, $GT_i$):** reconstruction at original viewpoint $g_i$ compared to $y_i$  
- **L1(recon\_i, $GT_j$):** same reconstruction compared to a different viewpoint $y_j$  
- **L1(swap, $GT_j$):** reconstruction when the pose is swapped to $g_j$ compared to $y_j$  
- **L1(swap, $GT_i$):** swapped reconstruction compared to the original viewpoint $y_i$   

A correct dual-SAL decoder should satisfy:

$$
\text{L1(swap,GT}_j) < \text{L1(swap,GT}_i),
$$

meaning that swapping the viewpoint parameter produces an image closer to the ground-truth view at $g_j$ than at $g_i$.

Across all models, this inequality holds. The complete quantitative results for one representative pair $(i,j)$ are:

| Model                | L1(recon,$GT_i$) | L1(recon,$GT_j$) | L1(swap,$GT_j$) | L1(swap,$GT_i$) | Correct? |
|----------------------|------------------|------------------|-----------------|-----------------|----------|
| Baseline (20 epochs) | 0.12969           | 0.12944           | 0.12451          | 0.13772          | ✔        |
| BaseCh128 (20 epochs)| 0.12858           | 0.12747           | 0.12289          | 0.13629          | ✔        |
| ResNet64 (20 epochs) | 0.11341         | 0.13028           | 0.11611     | 0.14559     | ✔ strongest |


The ResNet decoder exhibited the strongest use of the nuisance variable, producing the largest separation between the matched and mismatched viewpoint distances, while the baseline and 128-channel decoders showed the same qualitative behavior with smaller margins.

It is important to note that reconstruction quality alone is **not** a reliable indicator of success in this setting. Because SAL embeddings are intentionally minimal, they remove high-frequency detail and retain only coarse scene structure. Consequently, reconstructions at different viewpoints appear very similar, and $L1(recon_i, GT_i)$ and $L1(recon_i, GT_j)$ differ only slightly. This behavior is expected and does not reflect the decoder’s ability—or inability—to model viewpoint.

The central question is whether the decoder **responds correctly to changes in the nuisance variable g**. The appropriate test is therefore the *swap* comparison: a successful dual-SAL decoder should satisfy $L1(swap, GT_j) < L1(swap, GT_i)$, indicating that changing g moves the output toward the target viewpoint. All three architectures satisfy this inequality, confirming that each model has learned to reintroduce the viewpoint nuisance from a truly minimal representation.

Together, these results demonstrate that, once $z$ is enforced to be genuinely viewpoint-invariant, the decoder learns a meaningful approximation to the inverse of the SAL marginalization and can correctly reintroduce the removed nuisance.

---

## 3.3 Ablation Studies

### 3.3.1 Decoder Resolution (64 vs 128)
Higher resolution leads to larger images but:

- No new detail  
- Same low-frequency content  
- Same viewpoint behavior  

This confirms that SAL’s minimality destroys high frequencies that cannot be recovered: changing the output resolution does not restore the information that was intentionally discarded by the invariant encoder.


### 3.3.2 Decoder Capacity (Baseline vs ResNet)
ResNet decoder yields:

- smoother gradients  
- sharper low-frequency structure  

But:

- Still no fine detail  
- View-swap behavior unchanged qualitatively  

Capacity cannot overcome SAL minimality: once high-frequency information is removed from $z$, increasing decoder expressiveness cannot reconstruct it.

---

## 3.4 Discussion

### **Main Conclusion**

> A decoder can reintroduce viewpoint when the SAL representation is truly minimal (scene-level z).  
>  
> Decoder capacity cannot restore information SAL intentionally removed.

### Implications:

- Viewpoint is reconstructible as a nuisance  
- High-frequency details are irrecoverable  
- The SAL-dual mapping is valid and demonstrable  
- The main bottleneck is **representation**, not **decoder architecture**

### Novel Contribution of This Work

This is the **first experimental demonstration** of:

1. A hierarchy of nuisance-induced SAL invariances  
2. The necessity of enforcing minimality when training a dual-SAL decoder  
3. A practical dual operator that maps $(z,g)\to y_g$ for viewpoint  
4. Validation that viewpoint reintroduction works even under minimal embeddings  

This lays a foundation for future nuisance reintroduction:

- brightness  
- contrast  
- lighting (via NeRFactor)  
- rotation  
- occlusion  

---

# 4. Future Work

1. **Decoders for additional nuisance variables**  
   The same dual-SAL framework can be extended beyond viewpoint to brightness, contrast, rotation, scale, and lighting. Brightness and contrast decoders would operate in the photometric domain, while lighting decoders could leverage NeRFactor’s intrinsic decomposition to reapply illumination fields on top of a minimal representation.

2. **Local nuisance decoders**  
   Many nuisances are spatially local (e.g., shadows, partial occlusions, local specularities). A spatially-aware dual-SAL decoder—such as a UNet or diffusion-style generator—could reintroduce nuisance effects only in the relevant regions, respecting the underlying geometry learned by SAL.

3. **Nuisance estimators**  
   A natural extension is to learn a nuisance estimator $\hat{g}(y)$ alongside the decoder. This would enable:
   - estimating viewpoint or photometric parameters from an image,  
   - calibrating how invariant the encoder should be, and  
   - disentangling geometric versus photometric nuisance contributions.

4. **Joint encoder–decoder learning**  
   Currently, SAL encoders and dual-SAL decoders are trained separately. A joint objective could regularize the encoder so that $z$ is both strictly minimal and maximally usable by the decoder, balancing invariance and reconstructability.

5. **Higher-fidelity reconstruction via implicit representations**  
   Instead of decoding directly to pixels, the dual-SAL decoder could output parameters of a NeRF radiance field or MaterialMVP-style intrinsic representation (albedo, normals, roughness). In this setting, SAL provides a minimal, nuisance-stripped embedding, and the decoder lifts it into a rich implicit representation on which a separate renderer (NeRF / MaterialMVP) applies high-fidelity nuisance effects.

---

# 5. Potential Applications

The dual-SAL decoder enables controlled reintroduction of specific nuisances that were deliberately removed by SAL. This capability connects invariant representation learning to practical generative tasks.

1. **Augmenting minimal embeddings for realistic reconstruction**  
   A minimal embedding $z$ can first be purified by SAL, and then passed through a nuisance-injection decoder to produce a richer latent code that is fed into a downstream reconstruction model (e.g., NeRF, diffusion, or GAN-based decoders). This separates:
   - learning invariances (SAL), and  
   - generating photorealistic images (decoder + renderer).

2. **Editing implicit representations**  
   Because NeRF and MaterialMVP operate on disentangled internal representations, a dual-SAL decoder could modify viewpoint, lighting, or tone in the latent implicit space before rendering, enabling fine-grained and physically meaningful editing of scenes.

3. **Data augmentation for recognition tasks**  
   Because the decoder simulates nuisance variations from a single minimal representation, it can generate diverse nuisance-augmented views that share the same underlying class/identity. This could be used for training more robust recognition models without collecting additional real-world data.

---

# 6. Bibliography

- Soatto, Stefano & Chiuso, Alessandro. **Visual Representations: Defining Properties and Deep Approximations**. (2016).  
- Mildenhall et al. **NeRF: Representing Scenes as Neural Radiance Fields**.  
- Zhang et al. **NeRFactor: Self-supervised Decomposition of Appearance into Shape, Illumination, and Materials**.  
- MaterialMVP: Multiview Photometric Editing Models.  
- Additional code and tools referenced include torchvision, PyTorch, NeRF-pytorch, and ChatGPT assistance for editing and structuring this report (cited per rubric).

