# Transition Model pipeline

## Phase 1: Centered Latent Autoencoder.
### Vanilla AE
This step has been already implemented. We generate a "good" latent representation from our single-cell data, that can be reconstructed into the original state. Later we will work in this latent encoded state. The formulation is:  
$Z = E_{\theta}(X)$,  
where Z is the latent representation of single-cell gene expression vector X, $E_{\theta}$ is the encoder network parametrized by $\theta$ parameters. As an autoencoder we will encode non-perturbed and perturbed data into latent space and then reconstruct them to real space.  
$Z = E_{\theta}(X), \hat{X} = D_{\phi}(Z)$  
This pipeline is evaluated by a mean squared error loss: $L_{MSE}$.

### Class preserving.
When we create latent representation vectors, the class information can be mashed up, vanish, since the model is not told to keep that information.My approach to regularize the latent space by a GAN-like critic. For every gene, we also know their class $C$, which we can predict in latent space.

$\hat{C}_i = CL_{\Psi}(Z_i)$  
And then calculate the classification loss:
$L_{class} = CE(\hat{C}_i,C_i)$  
We add this loss to the original loss, therefore the model needs to learn class preserving representation.

### Centroid generation
Simple autoencoders doesn't regularize their latent space, one solution to this problem is the Varriational Autoencoder pipeline. I will not go too much into detail, but I had to realize that approach will not work in this case.Unfortunatelly class preserving information, does not mean distinguishable latent clustering. 
We added Centering loss which enforces to create latent space where classes are in separable clusters.
!["umap-not-centered"](umap_Epoch_0.png)  
!["umap-centered"](umap_Epoch_29.png)  
Note: First image does not show a full training, but in a fully trained model the results are similar. This is just an illustration of how unclustered latent space looks like.

### Final Loss
The whole loss function is made up from these main components and some minor regularizations, so the weights will not explode.

## Phase 2: Perturbation modelling.
Originally I thought to use an embedding model to embed perturbation effects into latent space as a condition and process it with a latent transition transformer model. From my current understandning perturbations can have different range of silencing effect, from small to total gene expression shut down. Latent embedding could be usefull and maybe able to model the transition effects.  

Currently desingning alternative approaches.  