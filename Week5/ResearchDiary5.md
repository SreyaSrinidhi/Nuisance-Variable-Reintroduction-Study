**1. Briefly summarize your knowledge of the area you are studying as of last week, and your plans for the current week from Q7 of your last diary. If this is the first diary write "Not Applicable."**

Answer: Last week, I wrapped up my experiments on the effect of kernel size and the Sampling and Anti-Aliasing Layer (SAL) on invariance and selectivity in CNNs. My plan for the current week, as stated in my previous diary, was to move beyond SAL and begin exploring how nuisance marginalization and minimal sufficient representations from Soatto & Chiuso (2016) connect to two broader areas: (1) diffusion-based image regeneration models and (2) interpretability of visual features.

This week, I followed through on the first direction—focusing on image regeneration—by reading NeRFactor and MaterialMVP, two recent works that disentangle intrinsic scene properties (geometry, albedo, BRDF) from nuisance factors like lighting and visibility. Building on the theoretical foundation from Soatto & Chiuso, I developed a new idea for a nuisance-conditioned generative renderer that learns a minimally sufficient 3D canonical representation and then re-applies learned nuisance transformations (lighting, pose, viewpoint, etc.) to regenerate images.

In short, I transitioned from analyzing convolutional invariance empirically to formulating a mathematical and generative framework that explicitly learns how nuisances act on invariant 3D structures—laying the groundwork for my next stage of research.

**2. List all resources you have read or looked at this week, along with the time you spent on each one (to the nearest 1/2 hour is enough). For web pages, blog posts, videos etc, provide links and titles. For papers, provide links and citations. For AI tools, provide a transcript of your session in a separate file in the Week_n folder, and list the file name (as a link) here.**

Answer: This week, I decided to move forward with looking at nuisance variables form a lens of image regeneration. The paper's leading to this decision is listed below:
* MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion, [Web Link](https://arxiv.org/pdf/2503.10289v1), [Github Link](../Papers/MaterialMVP.pdf)
  * Time Spent: 2hrs
* NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination, [Web Link](https://arxiv.org/pdf/2106.01970), [Github Link](../Papers/NeRFactor.pdf)
  * Time Spent: 2hrs
* [AI Transcript](../Week5/Week%205%20AI%20Transcript.pdf)
  * Time Spent (discussion + compiling transcript): 2hrs
* Time Spent on writing research diary: 1 hour

**3. Summarize what you have learned this week from the resources above. Be clear, detailed and precise. It is ok to be uncertain about the content. Do not copy/paste content from any resource.**

Answer: From MaterialMVP and NeRFactor, I learned how recent methods have begun bridging physically-based rendering with neural representations to factorize appearance into interpretable 3D components such as surface normals, albedo, visibility, and BRDFs. NeRFactor, for example, recovers these 3D neural fields under a single unknown illumination and can relight objects under new conditions by explicitly modeling light visibility and spatially varying reflectance. MaterialMVP extends this direction by introducing multi-view diffusion guided by physically-based reflectance parameters like metallic and roughness, which improves material consistency and relighting accuracy.

Connecting this to Soatto & Chiuso (2016), I realized that these decomposed 3D fields correspond closely to nuisance variables in the SAL framework — illumination, viewpoint, reflectance, and visibility — but instead of marginalizing over them, these newer models learn their transformations explicitly. This means the network effectively internalizes how nuisance factors act on the invariant 3D structure of the scene. Together, these insights suggest that it may be possible to build a model that learns the functional behavior of nuisances and can later apply them to regenerate appearance from a minimally sufficient 3D representation.

**4. Describe any new ideas you may have had as you were studying the resources, and if you did any follow ups to investigate these ideas.**

Answer: Building on these ideas, I developed a conceptual framework for a model that learns during training how nuisance variables act on a minimally sufficient 3D representation, and then applies those learned transformations to regenerate images. The model would be trained end-to-end to disentangle canonical (invariant) structure from nuisance factors like lighting, pose, reflectance, and visibility, using physically-based constraints similar to those in NeRFactor and diffusion-based material priors like MaterialMVP.

Unlike standard inverse-rendering models, this design focuses on learning the relationship between nuisance variables and scene appearance — rather than merely removing or normalizing them. In practice, this could take the form of a nuisance-conditioned generative renderer, which applies these transformations to a canonical 3D mesh to simulate real-world observations. Conceptually, it merges Soatto’s invariance theory with physically-grounded image factorization, allowing both interpretability and controllable image generation.

I have not implemented the idea yet, but I began outlining the mathematical mapping between the SAL marginalization term and NeRFactor’s rendering equation, with the goal of formalizing nuisance transformation as a differentiable learned operator within this pipeline.

**Proposed Pipeline:**<br>
1. Invariant encoder: $f_{\phi}$: <br>
   $x \mapsto z_{inv} ={geom,albedo,metallic,roughness}$ <br>
Training-time constraints: albedo/material constancy across lights, geometry regularizers (eikonal/Laplacian), smoothness priors on MR maps.
2. Nuisance predictor $q_{\psi}(g|x)$: <br>
   Predict $g = (pose, illum, exposure)$. If you have supervision for any part of $g$, use it; otherwise amortize and regularize (low-order SH for light, limited exposure range).
3. Nuisance-conditioned generative renderer $R_{\theta}$: <br>
Takes $z_{inv} + g$ and renders/regenerates $x$. Implement as:
   * fast: neural renderer / path-approx (direct + soft shadows), or
   * flexible: conditional diffusion $(x_t | x_{t+1}, z_{inv}, g, t)$.
4. Re-addition of nuisances with possible ordering:
   Model group composition: $g=(g_{pose},g_{vis}, g_{light},g_{camera})$ with a fixed application order, e.g.
   $$ x \approx R_{\theta}(z_{inv}, g_{camera} \circ g_{light} \circ g_{vis} \circ g_{pose}) $$
   Enforce with a group-consistency loss:
   $$
   \mathcal{L}_{grp} = || R_{\theta}(z_{inv},g_2 \circ g_1) - R_{\theta}(\Pi(R_{\theta}(z_{inv}, g_1)), g_2) ||_1
   $$
   where $\Pi$ is an (optional) differentiable back-projection to keep everything on-manifold. This makes “order then compose” ≈ “compose then order”, i.e., you truly learn to re-add nuisances.

**5. If you implemented/ran any code or algorithms from the resources or while investigating any new ideas, describe what you did. Link to a jupyter notebook in your Week_n folder showing the runs. You may also include python files containing code in your Week_n folder. If so, their content should be described here. If you forked another repo or imported pre-built code, please provide a link. If an AI tool wrote part of the code, please provide a session transcript in the Week_n folder and link to it here. If any part of the code did not run or did not behave as expected, describe your best guess why and possible fixes.**

Answer: N/A

**6. Summarize any specific points of confusion, uncertainty or difficulty from your reading or implementation that arose from your readings or implementations this week. This can partly overlap with your answer (3).**

Answer: One major point of uncertainty this week was clarifying where the boundary lies between intrinsic and nuisance factors within the model. Initially, I treated metallic and roughness as potential nuisance variables, but after analyzing how they parameterize the BRDF and remain constant under lighting or viewpoint changes, I realized they are better categorized as intrinsic material properties.

I also spent time disentangling the roles of pose and viewpoint. Although both correspond to geometric transformations, pose refers to the object’s position and orientation in world coordinates, whereas viewpoint describes the camera’s transformation relative to the object and affects projection and foreshortening. This distinction becomes important when designing the composition order $g_{camera} \circ g_{light} \circ g_{vis} \circ g_{pose}$ for the nuisance model.

I am still clarifying the composition and ordering of nuisance transformations. Many pairs are non-commutative (e.g., SE(3) rotations, visibility vs pose, lighting vs visibility), so the order likely matters. My current hypothesis is to apply them as:
$$ g_{camera} \circ g_{light} \circ g_{vis} \circ g_{pose} $$
which matches the rendering physics (pose → visibility → light transport → camera projection), but I need to verify this with a commutator study:
$$ ||R_{\theta}(z, g_i \circ g_j) - R_{\theta}(z, g_j \circ g_i)|| $$

A second uncertainty is training strategy: whether to rely purely on SAL-style sampling of $g$ (using labeled or amortized $q_{\psi}(g∣x))$ versus trying to “reverse” the pipeline to discover nuisance codes. My current plan is to (a) learn an invariant canonical encoder with contrastive losses across different $g$ of the same object, (b) condition the decoder on $g$ for reconstruction/relighting, and (c) add a group-consistency loss to stabilize composition.

Finally, I’m uncertain how much multi-illumination, multiview, and pose diversity I will need for identifiability; I expect to start synthetically and then move to real data.

**7. List specific goals you would like to accomplish for next week and action items aligned with these goals based on your answers above. Be as specific as you can.**

Answer: 
1. Formalize the mathematical composition of nuisance transformations — derive clear equations for how pose, viewpoint, lighting, and visibility compose and act on the canonical mesh.
2. Implement a toy prototype using NeRFactor-style decomposition (geometry, albedo, BRDF, visibility) extended with metallic and roughness as intrinsic parameters. This will serve as the canonical $z_{inv}$.
3. Design the nuisance re-addition module, beginning with a simplified subset (e.g., lighting + pose) to test the feasibility of sequential nuisance application.
4. Develop a differentiable group-consistency loss, ensuring that $R_{\theta}(z_{inv},g_2 \circ g_1) \approx R_{\theta}(\Pi(R_{\theta}(z_{inv},g_1)),g_2)$
5. Begin drafting a short concept note summarizing how this model unifies SAL marginalization (for invariance) and conditional generative rendering (for nuisance application).

Action Items:
* Review relevant sections of the NeRFactor paper describing visibility and reflectance recovery, and the MaterialMVP diffusion conditioning strategy.
* Write pseudocode for the forward rendering pipeline showing how $z_{inv}$ and $g$ interact.
* Experiment with simple synthetic data (e.g., spheres under variable lighting) to visualize the effect of sequential nuisance composition.
* Draft mathematical notation and visual diagrams