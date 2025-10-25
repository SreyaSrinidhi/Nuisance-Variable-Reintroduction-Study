### 1. Briefly summarize your knowledge of the area you are studying as of last week, and your plans for the current week from Q7 of your last diary. If this is the first diary write "Not Applicable."

Answer: Last week, my goal was to study how kernel size interacts with the Sampling and Anti-Aliasing Layer (SAL) and how this affects the invariance–selectivity trade-off in convolutional neural networks. Specifically, I planned to run experiments comparing models with kernel sizes of 3×3, 5×5, and 7×7, each with and without SAL, and evaluate their performance under small geometric nuisances (translation, rotation, scale) and occlusion. I also planned to start setting up experiments for testing residual vs. plain blocks under similar nuisance conditions.

This week, I implemented the kernel size × SAL experiment using the Geometric Shapes dataset from Kaggle and built a controlled CNN architecture to isolate the effect of kernel size and SAL on model performance.

### 2. List all resources you have read or looked at this week, along with the time you spent on each one (to the nearest 1/2 hour is enough). For web pages, blog posts, videos etc, provide links and titles. For papers, provide links and citations. For AI tools, provide a transcript of your session in a separate file in the Week_n folder, and list the file name (as a link) here.

Answer: This week, I ran experiment to investigate the impact of SAL on different kernel sizes. This was based off the paper
* Visual Representations: Defining Properties and Deep Approximations, S. Soatto, A. Chiuso, ICLR 2016; [Web Link](https://arxiv.org/pdf/1411.7676v9), [Github Link](../Papers/Visual_Representations_defining_properties_and_deep_approximations.pdf)
* AI Tool: ChatGPT session for code design, debugging, and interpreting low-accuracy results
* Time Spent on Experiment: 5 hrs
* Time Spent on writing the research diary: 1 hr

### 3. Summarize what you have learned **this week** from the resources above. Be clear, detailed and precise. It is ok to be uncertain about the content. Do **not** copy/paste content from any resource.

Answer: This week, I learned how receptive field size and anti-aliasing interact in shaping invariance and selectivity in CNNs. Smaller kernels (3×3) tend to maintain sharper discriminative boundaries and perform better on classes that differ by edge geometry, while larger kernels (5×5 and 7×7) increase spatial smoothness but risk losing fine-grained shape distinctions—particularly when combined with SAL.

I also discovered that the training data augmentations themselves can create nuisance conflicts. Specifically, including rotation augmentation hurt performance because in this dataset, shape labels (e.g., square vs. rhombus vs. parallelogram) are orientation-dependent. After removing rotation and shear, validation accuracy increased substantially (from ~50–70% to expected 85–90%). This clarified that the choice of nuisance transformations must align with the label semantics of the dataset.

In short:

* SAL helps with invariance by smoothing before pooling, but too much smoothing (especially with large kernels) reduces selectivity.
* Kernel size influences the effective receptive field and determines how much local context the network averages over.
* Proper augmentation design is critical when studying transformation invariance empirically.

### 4. Describe any new ideas you may have had as you were studying the resources, and if you did any follow ups to investigate these ideas.

Answer: After discussing with Prof. Ray, I decided to explore two next directions that connect to the broader theme of transformations and invariance:
1. Interpretability and Explainability of CNNs — investigating how feature maps change under controlled nuisances and how receptive fields evolve layer-by-layer.
2. Diffusion Models for Image Generation — leveraging the “inversion” perspective of SAL and the idea of optimal visual representations for reconstructing or regenerating images under controlled nuisance conditions.

The diffusion direction also ties to my earlier third goal (“add-back nuisances”)—it could provide a way to reintroduce nuisance parameters (like orientation or illumination) in a generative model while maintaining invariance in the discriminative path.

### 5. If you implemented/ran any code or algorithms from the resources or while investigating any new ideas, describe what you did. Link to a jupyter notebook in your Week_n folder showing the runs. You may also include python files containing code in your Week_n folder. If so, their content should be described here. If you forked another repo or imported pre-built code, please provide a link. If an AI tool wrote part of the code, please provide a session transcript in the Week_n folder and link to it here. If any part of the code did not run or did not behave as expected, describe your best guess why and possible fixes.

Answer: I implemented a full training and evaluation pipeline in [Link to Jupyter Notebook](./experiment.ipynb).
Key details:
* Dataset: Kaggle Geometric Shapes Mathematics dataset (eight classes).
* Input size: 224×224×3.
* Models: Six CNNs (3×3, 5×5, 7×7 kernels × with/without SAL blur-pool).
* Framework: PyTorch (custom architecture KernelNet, parameter-matched across kernel sizes).
* Metrics: Validation/test accuracy, feature stability (to be extended next week).
Initially, I got low accuracies (≈0.39–0.73) due to inappropriate rotation augmentation. After disabling rotation and increasing epochs (to 40) with adjusted learning rate (1e-3), accuracy improved significantly.

### 6. Summarize any specific points of confusion, uncertainty or difficulty from your reading or implementation that arose from your readings or implementations this week. This can partly overlap with your answer (3).

Answer: 
* Initially, I was confused why model accuracy was so low despite a simple dataset. I learned this stemmed from label inconsistency under rotation augmentation, not from the SAL implementation.
* I also realized that in BlurPoolDown, the kernel size of the blur (fixed 5×5) is independent of the convolutional kernel size being tested (3×3, 5×5, 7×7). This separation is intentional—k=5 is for low-pass filtering, while K controls the learned receptive field.
  
### 7. List specific goals you would like to accomplish for next week and action items aligned with these goals based on your answers above. Be as specific as you can.

Answer: 
1. Explore next research direction beyond SAL:
Begin investigating how the concepts of nuisance marginalization and minimal sufficient representations from the Soatto & Chiuso paper connect to more recent approaches in: 
   1. Image regeneration and diffusion models — studying how diffusion processes can act as the inverse or reconstruction counterpart to SAL, where nuisance information is selectively added back during generation.

   2. Interpretability and explainability of CNNs — analyzing how SAL and receptive-field behavior influence feature attribution and visual reasoning, and whether explainability metrics (e.g., saliency, Grad-CAM, feature visualization) can reveal the invariance–selectivity structure proposed in the paper.
2. Survey state-of-the-art work in both directions:
   * For diffusion: focus on foundational works such as Denoising Diffusion Probabilistic Models (Ho et al., 2020) and newer inversion/conditional regeneration techniques.
   * For interpretability: review Feature Visualization by Optimization, Grad-CAM, and Network Dissection to understand existing attribution frameworks. 
3. Identify potential novelty and contribution space:
   * Brainstorm how SAL’s formalism (local marginalization of nuisances) could be incorporated into diffusion sampling or interpretable feature alignment.
   * Define a concrete experiment or theoretical angle where I can extend or modify an existing algorithm to demonstrate this connection.