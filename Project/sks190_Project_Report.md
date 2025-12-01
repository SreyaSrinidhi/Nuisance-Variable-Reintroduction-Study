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
 that explicitly reintroduces a chosen nuisance configuration g to reconstruct the corresponding image $y_g$. Thus, while SAL removes the dependence on $g$, our decoder $D_{\psi}$ maps back from the invariant representation to the space of nuisance-affected observations, realizing a practical dual to the SAL operation for the nuisance family consisting of pose, lighting and contrast.

## Results, Analysis, and Discussion

## Bibliography