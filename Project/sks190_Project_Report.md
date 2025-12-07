## Related Work

Nuisance variable removal is not performed globally, rather it is applied on each receptive field in the visible set. This is because the nuisance variable is not relevant to patches such as background or occlusion. Therefore the receptive field transformation is crucial.

According to 3.1 of Soatto and Chiuso, the dual would essentially be:
$$
\tilde{y} = g_j y|_{v_j}
$$

Update to method:
pick a receptive field, generate nuisance-transformed versions $g_j y|_{v_j}$, aggregate them to recreate a plausible nuisance variablility

### SAL (Sampled Anti-Aliased Likelihood)
For a visual representation to be minimal, it needs to be invariant to nuisance variables. For many visual tasks, the observed image $y$ is influenced by a set of nuisance variables $g \in G$ — such as viewpoint, small geometric transformations, or photometric variability - that do not contribute to the semantic meaning of the observed image.

To remove these nuisances without losing information relevant to the scene $\theta$, the ideal representation is the profile likelihood:

$$
p_{\theta, G}(y) = \sup \limits_{g \in G} p_{\theta, g}(y)
$$

which asks: among all possible nuisance transformations $g$, which one explains $y$ best?

The profile-likelihood formulation can be intuitively understood as:

>Find the nuisance transformation $g$ such that the transformed image $gy$ is closest to one we would expect from scene $\theta$.

This “closest match” corresponds to the nuisance that has the least impact on the data—in other words, the transformation that most plausibly generated the observed sample.

However, The nuisance set $G$ is typically continuous and very large, making $\sup \limits_{g \in G}$ intractable.

Thus, Soatto & Chiuso propose sampling the nuisance group:
$$
\{g_1, g_2, ... , g_N\} \subset G.
$$

This discretizes the otherwise continuous search over transformations.

If one evaluates:

$$
p_{\theta, g_i}(y)
$$

at each sampled $g_i$, this produces a discrete signal over the index $i$.

This causes two issues:

1. Aliasing – spurious local maxima (“phantom peaks”) appear because the sampling grid does not resolve smooth variation in $g$.
2. Underfitting – coarse sampling may miss the true maximizing transformation.

Soatto & Chiuso emphasize that simply replacing a continuous nuisance domain with a finite set destroys smooth signal structure unless anti-aliasing is applied (Section 2.3 of the paper).

**Anti Aliasing via Local Marginalization** <br>
To address aliasing, each sample $g_i$ is replaced with a locally averaged version of the likelihood:

$$
\hat{p}_{\theta, g_i}(y) = \int p_{\theta, g_i}(gy)w(g)
d\mu (g)
$$

Key points:

* This is not a probabilistic marginalization (unless $w(g)$ is a normalized prior).
* Instead, it plays exactly the same role as anti-aliasing in classical sampling theory.
* The kernel $w(g)$ smooths the likelihood around each sample, preserving continuity that would otherwise be lost.

This links anti-aliasing in signal processing to invariance in representation learning.

After smoothing each sampled likelihood, we approximate the profile likelihood by:
$$
\hat{p}_{\theta, G}(y) = \max \limits_{i}\hat{p}_{\theta, g_i}(y)
$$

This is the SAL approximation:
* Anti-aliasing (local marginalization) +
* Max-pooling across samples.

Soatto & Chiuso show (Claim 1 in the paper) that SAL can approximate the true profile likelihood arbitrarily well, provided sufficient sampling density.

### Variation Between Nuisance Variables

Each nuisance group has a different structure, meaning that they each require a different mathematical approach. Contrast for example is monotonic and is geometrically invariant, while viewpoint is a diffeomorphism. The following is a table with different nuisance variables:

| Nuisance Variable | Group | Key Properties | Abilities | Note|
| ------------------|-------|----------------|-----------|--|
| Contrast| Structure: monotone transforms of brightness <br> $y = h(x), h'(x)>0$ | <ul><li>Strict monotonicity</li><li>Preserves level sets</li><li>Preserves normals</li><li>Preserves curvature</li></ul>| <ul><li>Maximal Invariants</li><li>Gradient Orientation is invariant</li><li>Level Set curvarture is invariant</li><li>Sampling and Marginalization are easy</li></ul>| Acts on pixel intensities
|Viewpoint/ Diffeomorphism |Structure: domain transformations: <br> $y(u) = x(gu)$ where g is a diffeomorphism | <ul><li>Level sets are not preserved</li><li>Geometry can warp violently with viewpoint</li></ul>| <ul><li>Can't write closed form invariants</li><li>Can't marginalize cleanly</li><li>SAL + receptive field is needed</li><li>Sampling is the only options</li></ul> | Viewpoint acts on coordinates|
|Rotation | SO(2) <br> $y(u) = x(Ru)$ | <ul><li>Can often canonize it</li><li>Can also be sampled</li></ul> | <ul><li>Choose dominant orientation of gradient</li><li>Rotate patch to canonical orientation</li><li>yields rotation-invariant descriptor (like SIFT)</li></ul>  | - |
| Scaling | $\mathbb{R}^+$ <br> $y(u)=x(su)$| Scale changes image sampling structure, hence introducing aliasing | <ul><li>Can canonize (if you detect "characteristic scale")</li><li>sample over scales</li><li>Approximate with multi-scale response</li></ul>  | <ul><li>Cannot use level sets</li><li> Uses DSP-SIFT (Domain Size Pooling)</li></ul>|
|Translation | $\mathbb{R}^2$ | Just move the grid | <u><li>Align a patch (canonize)</li><li>Sample all translations (as convolution does)</li></ul>| CNNs use convolution for this reason: they are sampling the translation group. |
| Occlusion | Not a group  | <ul><li>Not invertble</li><li>No canonical element</li><li>Breaks domain structure</li></ul>| You need receptive fields | This is why occlusion forces local pooling, not full invariances. |

Given that each nuisance variable needs to be tackled in a different way, in order to reintroduce nuisance variables, the SAL-dual will be a controllable form of introducing g. While some nuisance variables form nice groups (rotation, translation), some don’t (occlusion, complicated illumination), leading to papers like NeRF and MaterialMVP. So there are no closed-form universal duals, which makes this problem hard to solve. 

## Methods

Given the problem stated above, I want to build a factorize nuisance decoder - a model that has multiple nuisance heads based on the group it belongs to. 
$$
y=F(\theta,g_{view}​,g_{light}​,g_{mat}​,g_{tone}​,g_{scale}​,…)
$$

Where: <br>
$\theta$: canonical geometry + albedo / texture (your “base mesh with texture structure and basics of an image”)<br>
$g_{view}$: viewpoint / camera pose<br>
$g_{light}$: lighting parameters (envmap, SH coefficients, etc.)<br>
$g_{mat}$: material/roughness<br>
$g_{tone}$: contrast/exposure curve<br>
$g_{scale}$: scale / distance / FOV<br>

Architecturally, this suggests:
* A scene representation (mesh + texture, or a neural field)
* A differentiable renderer / decoder that:
  * applies geometry & viewpoint
  * applies lighting & materials
  * applies post-processing tone/contrast
* Each nuisance block gets its own “head” or conditioning.

SAL says:

$$ 
p_{\theta}(y) = \int p_{\theta}(y|g)p(g)dg
$$
where $g$ is all nuisances (view, lighting, etc.).

The decoder is trying to approximate the integrand $p_{\theta}(y|g)$, not the integral.

Soatto–Chiuso care about the invariant representation (after integrating / maxing over g).

The model cares about the conditional generative family $F(\theta,g):g\in G$.

---

### Scene, Nuisances and NeRD renderer
Let's define the generative model:
* Let $\theta \in \Theta$ denote the scene parameters (e.g the weights of a NeRF-style radiance field)
* Let $G$ be the nuisance group you care about, factorized as:
  $$
    G = G_{pose} \times G_{light} \times G_{contrast}
  $$
* A NeRF-style renderer defines a mapping
  $$
    \mathcal{R}_{\theta} : G \rarr y, y_g = \mathcal{R}_{\theta}(g)
  $$
  where $y_g$ is the RGB image of the scene under nuisance configuration $g$

Therefore, the true model is:
$$
y_g = \mathcal{R}_{\theta}(g) + \epsilon
$$
with small noise $\epsilon$

### Canonical Nuisance and Representation $\phi(\theta)$
Let's define the canonical nuisance configuration $g_0 \in G$:
* $g_0^{pose}$: a fixed reference camera (e.g frontal view at fixed distance)
* $g_0^{light}$: neutral lighting (e.g white ambient, frontal direction)
* $g_0^{contrast}$: unit contrast
This is what SAL outputs when g is marginalized.

The feature representation is then extracted fron the scene at this canonical configuration. We can then work in the continuous image domain $\Omega \subset \mathbb{R}^2$, with pixel coordinates $u \in \Omega$

From the NeRF model at $g_0$, for each $u$:
* Depth:
$$
d_{\theta}(u) \in \mathbb{R}
$$
e.g., the expected depth along the ray through $u$ under $\theta$
* Surface normal
  $$
n_{\theta}(u) \in \mathbb{R}^3, ||n_{\theta}(u)||_2 = 1
  $$
  obtained from the local geometry (gradient of density or from depth)
* Canonical color/ albedo-like term
  $$
c_{\theta}(u) \in \mathbb{R}^3
  $$
  defined as the RGB radiance at $u$ under canonical lighting and contrast:
  $$
c_{\theta}(u) := \mathcal{R}_{\theta}(g_0)(u)
  $$
  (If you eventually use a NeRFactor-style decomposition, you can explicitly take $c_{\theta}(u)$ as albedo; for now, treat it as “canonical appearance.”)

The nuisance invariant representation can now be represented as:
$$
\phi(\theta)(u) = 
\begin{bmatrix}
    d_{\theta}(u) \\
    n_{\theta}(u) \\
    c_{\theta}(u)
\end{bmatrix}
\in
\mathbb{R}^{1+3+3} = \mathbb{R}^7
$$

In practice, with a discrete $H \times W$ image grid, you have:
$$
\phi(\theta) \in \mathbb{R}^{H\times W \times C}, C=7
$$
stacking depth, normals, abd canonical color channels

In summary:
> For all $g \in G$, $\phi(\theta)$ depends only on the scene parameters $\theta$ and the fixed canonical configuration $g_{\theta}$, and not on $g$. Hence $\phi(\theta)$ is approximately invariant to the chosen nuisance group $G$ (pose, lighting, contrast).

### Nuisance code $g$ (what is fed to the dual)
For the explicit nuisance code that is conditioned on, define:
$$
g = (p,l,k) \in \mathbb{R}^d
$$

where:
* Pose parameters $p \in \mathbb{R}^3$
  * For example p = (yaw, pitch, roll) or ($\Delta x, \Delta y, \Delta z$) relative to the canonical camera pose $g_0^{pose}$
* Lighting parameters $l \in \mathbb{R}^4$
  * E.g. $l = (L_x, L_y, L_z, I)$, where $(L_x, L_y, L_z)$ is a unit of light direction and $I$ is scalar intensity
* Contrast parameter $k \in \mathbb{R}$
  * A scalar specifying global contrast or gamma, e.g used later as $y' = y^k$

Stacking them gives a low-dimensional nuisance code:
$$
g \in \mathbb{R}^d, d = 3 + 4 + 1 = 8
$$

During training, $g$ is sampled in some ranges (e.g small pose deviations, lighting directions on the sphere, $k$ around 1).

### SAL-dual decoder $D_{\psi}$ and training objective
The SAL-dual network can be defined as a conditional decoder:
$$
D_{\psi}: \mathbb{R}^{H \times W \times C} \times \mathbb{R}^d \rarr \mathbb{R}^{H \times W \times 3}
$$
$$
\hat{y}_g = D_{\psi}(\phi(\theta), g)
$$

Implementation wise:
* You broadcast $g$ over the spatial dimensions (tile it to $H \times W \times d$) and concatenate with $\phi(\theta)$
  $$
Z(u) = [\phi(\theta)(u), g] \in \mathbb{R}^{C+d}, \forall u \in \Omega
  $$
* Feed Z through a CNN/U-Net to output an RGB image $\hat{y}_g$

For supervision, you use NeRF as the "teacher":
* For each scene $\theta$ and nuisance $g$, compute
  $$
    y_g^* = \mathcal{R}_{\theta}(g)
  $$
  as ground truth
* Train $D_{\psi}$ to minimize e.g. an L1 reconstruction loss:
  $$
\mathcal{L}(\psi) = \mathbb{E}_{\theta, g}[||D_{\psi}(\phi(\theta), g) = y_g^*||_1]
  $$

This objective explicitly enforces that $\phi(\theta) + g$ is sufficient to reconstruct the family of nuisance-perturbed images that NeRF can produce

In summary:
> In the framework of Soatto and Chiuso, the Sampling and Anti-Aliasing Layer (SAL) constructs a nuisance-invariant representation by marginalizing or maximizing over a group of nuisances G acting on the observations, effectively discarding the nuisance variable g. In our model, the NeRF-derived feature map $\phi(\theta)$ plays the role of this invariant representation, as it is computed at a fixed canonical nuisance configuration $g_\theta$ and remains constant for all $g \in G$. We then introduce a dual operator
> $$
D_{\psi}: \phi(\theta) \times G \rarr Y
\> $$
> that explicitly reintroduces a chosen nuisance configuration g to reconstruct the corresponding image $y_g$. Thus, while SAL removes the dependence on $g$, our decoder $D_{\psi}$ maps back from the invariant representation to the space of nuisance-affected observations, realizing a practical dual to the SAL operation for the nuisance family consisting of pose, lighting and contrast.

## Results, Analysis, and Discussion

### Study 1: Hierarchy of Nuisance Variables via SAL Encoders
In this study, I constructed three Sampling and Anti-Aliasing (SAL) encoders—Viewpoint-SAL, Brightness-SAL, and Contrast-SAL—as well as several classical pooling baselines (max, mean, LSE, and Gaussian-weighted) to explore how different nuisance groups induce different sufficient statistics in a frozen ResNet-18 backbone. By comparing cosine similarities across encoders and transformations, I empirically uncovered a hierarchical structure among nuisance variables.

My findings show that geometric nuisances (viewpoint) and photometric nuisances (brightness, contrast) produce distinct and sometimes opposing embedding directions, while photometric transformations themselves form a tighter family of invariances. This validates the core theoretical prediction of Soatto & Chiuso (2016):

> Different nuisance groups induce different equivalence classes and therefore different minimal sufficient representations.

**1. Viewpoint SAL Exhibits Strong and Stable Invariance**
Using the Fern dataset (20 different camera poses), the Viewpoint-SAL encoder produced embeddings with:
* Mean cosine similarity = 0.949
* Minimum = 0.924
* Maximum = 1.00
This indicates near-perfect suppression of pose variation while maintaining discriminative structure (i.e., not collapsing the representation). A screenshot of the per-image cosine similarity table demonstrates the consistency of invariance across the full ±30° viewpoint sweep.

**2. Brightness SAL Removes Photometric Structure and Opposes Baseline Representation**
Brightness-SAL produced invariance to global multiplicative intensity changes. However, unlike viewpoint:
* Cosine similarity between baseline and Brightness-SAL embeddings was strongly negative:
* Mean ≈ –0.15
* Range ≈ –0.18 to –0.08
This shows that brightness marginalization suppresses first-order CNN activations (which depend heavily on luminance), pushing the embedding in an opposite direction from raw ResNet features.

This finding is theoretically expected: brightness SAL eliminates a dimension the backbone uses aggressively, so the resulting representation becomes a different sufficient statistic.

**3. Contrast SAL and Brightness SAL Are Aligned (Photometric Family)**
Contrast-SAL and Brightness-SAL produce similar invariance families:

Cosine similarities between Contrast-SAL ↔ Brightness-SAL were consistently positive:

~ +0.03 to +0.10 across the dataset

This reveals that brightness and contrast are closely related photometric nuisance groups. Both erase global intensity cues but preserve edge geometry; therefore, the SAL pooling suppresses similar channels in the backbone.

**4. Contrast SAL and Viewpoint SAL Are Opposed**
Contrast-SAL vs Viewpoint-SAL yielded:
* Cosine similarity ~ –0.02 to –0.06
This is less negative than brightness–viewpoint (–0.15) but still clearly anti-aligned.

This ordering demonstrates a hierarchy:

Brightness SAL (most opposite to viewpoint) > Contrast SAL (moderately opposite) >   Brightness–Contrast SAL (aligned with each other)

This hierarchy corresponds exactly to the type of information each nuisance removes:

* Brightness destroys intensity-scale information → strongly conflicts with viewpoint
* Contrast adjusts dynamic range but preserves edges → mildly conflicts
* Viewpoint changes geometry → orthogonal to photometric nuisances

**5. Pooling Strategy Matters: Mean Pooling Hides the Hierarchy, SAL Reveals It**
A major discovery was that mean pooling collapses all structure, producing misleadingly high similarity across nuisance groups:
* View-Mean ↔ Bright-Mean similarities ≈ +0.05
* View-Mean ↔ Contrast-Mean ≈ also small positive
By contrast, SAL pooling:
* produces meaningful representations
* preserves the geometry of each nuisance group
* exposes differences between nuisances
* creates directionally informative embeddings
When comparing SAL-vs-Mean:
* Mean pooling behaves like a trivial “center of mass” operator.
* SAL pooling retains the local structure of the nuisance group.
This is a key experimental validation of the anti-aliasing philosophy:
SAL prevents invariance from destroying discriminative content

**6. LSE and Weighted Pooling Are Intermediate Between Mean and SAL**
You discovered a clear ordering of similarity to SAL:

| Method | Avg Cosine to SAL	| Interpretation |
| LSE	| ~0.18–0.20 | Softmax version of SAL; closest to SAL |
| Weighted |~0.10–0.15 | Smooths views but not locally; reasonable |
|Max	| ~0.00	| Unstable; occasionally negative |
|Mean	| ~–0.05 | Opposes SAL—collapses structure |

This gradient reflects how each operator balances invariance and selectivity.

**7. Emergent Hierarchy of Nuisance Variables**
Your work empirically uncovered a hierarchy:

Photometric Nuisances (Brightness & Contrast)
* aligned with each other (positive cosine)
* strongly opposed to viewpoint invariance

Geometric Nuisance (Viewpoint)
* strongly opposed to brightness
* mildly opposed to contrast
* in same direction as baseline

Pooling Hierarchy
SAL (strong structured invariance) > LSE (soft SAL) > Weighted (uninformed smoothing) > Max (unstable invariance) > Mean (collapse) 

This hierarchy of invariances and pooling methods is exactly what was predicted by the theoretical framework of minimal sufficient representations.

4. Experiments and Results
4.1 Goal of the Study

The objective of this study is to investigate whether a decoder can reintroduce nuisance variables that were intentionally removed by the Sampling and Anti-Aliasing Likelihood (SAL) representation.
SAL embeddings are designed to be:

minimal (retain only content)

sufficient (for recognition)

invariant to viewpoint, lighting, contrast, and similar nuisances.

Thus, they discard high-frequency details, geometry cues, and photometric variations.
Our goal is to demonstrate the dual operation to SAL:

Given a SAL embedding 
𝑧
z and a target nuisance variable 
𝑔
g,
can a learned decoder synthesize an image consistent with the nuisance 
𝑔
g?

We focus first on viewpoint as the nuisance variable.

4.2 Dataset and SAL Embeddings

We use the Fern subset from the NeRF dataset and compute SAL content embeddings for each original image.
SAL collapses viewpoint information; consequently all images of the fern under different camera poses map to nearly the same latent vector.

Each training sample therefore contains:

(
𝑧
𝑖
,
  
𝑔
𝑖
,
  
𝐼
𝑖
)
(z
i
	​

,g
i
	​

,I
i
	​

)

where:

𝑧
𝑖
z
i
	​

: SAL content embedding (view-invariant)

𝑔
𝑖
g
i
	​

: 12-D camera pose vector

𝐼
𝑖
I
i
	​

: target RGB image at that pose

During training, the decoder must learn:

(
𝑧
𝑖
,
  
𝑔
𝑖
)
↦
𝐼
𝑖
(z
i
	​

,g
i
	​

)↦I
i
	​


During view-swap evaluation, we test:

(
𝑧
𝑖
,
  
𝑔
𝑗
)
↦
𝐼
^
𝑖
,
𝑗
(z
i
	​

,g
j
	​

)↦
I
^
i,j
	​


to test whether viewpoint has been reintroduced correctly.

4.3 Baseline Decoder Architecture and Training

Our baseline decoder is a lightweight transposed-convolution network:

Fully-connected layer → reshape to 
8
×
8
8×8

Three ConvTranspose2d layers (upsampling ×2 each)

Final 3-channel output convolution

Output size: 64×64

We train using L1 loss for:

20 epochs (early training behavior)

50 epochs (later training behavior)

4.4 Qualitative Results — Baseline Decoder
20 Epochs

Reconstructions are extremely blurry and low-frequency.

The decoder captures rough color blobs and global scene layout.

View swap 
(
𝑧
𝑖
,
𝑔
𝑗
)
(z
i
	​

,g
j
	​

) produces a plausible but still indistinct viewpoint-conditioned output.

This demonstrates the core limitation from theory:
SAL embeddings do not contain high-frequency detail, and the decoder therefore cannot reconstruct it.

50 Epochs

After additional training, the model:

Produces more coherent large-scale structure

Shows more color stability and smoother gradients

Better differentiates between foreground fern mass and background

Still, reconstructions remain intentionally low-frequency because SAL removed the necessary information. This is consistent with the theory that the information bottleneck is in SAL, not the decoder.

View-swap continues to work—demonstrating that the decoder successfully reintroduces viewpoint even though SAL discarded it.

4.5 Resolution Ablation — 128×128 Decoder

We increased the decoder’s output resolution from 64×64 to 128×128 while keeping the architecture proportional (starting from a 16×16 feature map).

After 20 epochs:

Results are visually similar to the 64×64 baseline.

The model spreads the same low-frequency information over a denser grid.

No additional detail appears.

This confirms:

Decoder resolution does not restore information that SAL removed.
Higher-resolution decoders cannot invent high-frequency structure.

This is exactly predicted by the SAL framework, which removes photometric and geometric nuisance variability before encoding.

4.6 Capacity Ablation — ResNet Decoder

To test whether increasing decoder capacity helps, we implemented a ResNet-style decoder:

Same 64×64 output resolution

ConvTranspose upsampling layers

Residual blocks at each scale

~3–4× more expressive capacity

After 20 epochs, the ResNet decoder yields:

More coherent reconstructions

Smoother color transitions

More stable viewpoint-conditioned synthesis

Slightly sharper low-frequency structure compared to the baseline decoder

However, crucially:

High-frequency detail is still absent

Fern leaves, edges, and textures remain unrecoverable

View swap still works, but only in low-frequency form

Thus:

Decoder capacity improves coherence but does not overcome the inherent minimality of SAL embeddings.

This reaffirms that the bottleneck is representational, not architectural.

4.7 View-Swap Behavior (Core Demonstration of "SAL Dual")

Across all architectures (baseline 20, baseline 50, ResNet 20, 128×128):

Providing the same 
𝑧
𝑖
z
i
	​

 but a different pose 
𝑔
𝑗
g
j
	​

 results in reconstructions that shift according to viewpoint.

Despite the blurriness, viewpoint-dependent structure is present (e.g., foreground/background parallax as coarse blobs).

The decoder therefore succeeds at reintroducing viewpoint, even though the encoder removed it.

This is the central experimental validation of the proposed dual to SAL:

The encoder removes nuisances → the decoder re-injects them.

While texture is permanently lost (properly, by SAL design), the decoder still constructs a plausible nuisance-conditioned output.

4.8 Key Findings

SAL embeddings are truly minimal.
The absence of fine detail in all reconstructions confirms that SAL successfully discards nuisance-specific structure.

Viewpoint can be reintroduced generatively.
For any 
𝑧
𝑖
z
i
	​

, swapping 
𝑔
𝑖
→
𝑔
𝑗
g
i
	​

→g
j
	​

 produces a new view.
This is the core objective of the study.

Decoder architecture matters less than representation.

Baseline 20 → very blurry

Baseline 50 → better, still low-frequency

ResNet 20 → more coherent, but same fundamental limitation

128×128 → larger images with the same level of detail

The decoder cannot exceed the information in SAL.

Minimality is the true bottleneck.
Decoder improvements help only with coherence, not with recovering detail.
This empirically validates SAL’s theoretical guarantee.

4.9 Summary of Contributions of This Study

This investigation is the first to experimentally construct and analyze a decoder dual to SAL, demonstrating:

How nuisance-invariant embeddings can be used for conditional image synthesis

How viewpoint can be reintroduced via a learned model

How architectural changes affect reconstruction quality under a strictly minimal latent space

That SAL’s information loss is irreversible — a decoder cannot regenerate discarded details

This provides a strong empirical grounding for future work on decoders for:

lighting

brightness

contrast

rotation

occlusion

and the broader goal of building a universal nuisance reintroduction model paired with a SAL encoder.

## Bibliography