# obesity_challenge_broad
https://hub.crunchdao.com/competitions/broad-obesity-1

## 1. Resources:
    1. Youtube 80 min long lecture. 
        https://www.youtube.com/watch?v=-qOs32IFDvY&list=PLlMMtlgw6qNi-WSkY-UiJMvO2TTa0T4Ga&index=6
    2. https://hub.crunchdao.com/competitions/broad-obesity-1
    3. quick starters:
        https://hub.crunchdao.com/competitions/broad-obesity-1/resources/quickstarters
    4. the paper :
        https://docs.google.com/viewerng/viewer?url=https://crunchdao--competition--production.s3.eu-west-1.amazonaws.com/competitions/broad-obesity-1/documents/broad-obesity.pdf
    
## 2. Leaderboard:
    https://hub.crunchdao.com/competitions/broad-obesity-1/leaderboard
## 3. Submission: 
    https://hub.crunchdao.com/competitions/broad-obesity-1/models/melancholic-regina/electrical-damselfly  
    
## Pipeline
This is the proposed pipeline by Bence. We don't have to strictly follwo this, but it would be good if we keep ourselves to a planned approach. If you have ideas what to change we can do that.
### Data representation
Find a nice way to represent these large vectors. Default: split into smaller sequences-> baseline tokens-> they can be handled as a sentence of texts. 

### State Regressor
We need to build a model that can approximate new state based on the perturbation. My idea would be to first train a VAE system for representing non-perturbed genes. In this case the important part is the low dimensional probability distribution representation, the types of blocks used for it might be still in question (possibly transformer encode, but based on data representation it can vary). To further improve on this we can build a VAE-GAN, where the VAE needs to generate new perturbations from the learnt distribution, while the GAN needs to address if it's a non-perturbed or a perturbed gene (if we can learn it at all). But this idea may be more usefull in the next stage. The represented latent vectors can be put through a decoder module (probabbly transformer decoder) and cross-attentioned with the perturbation vectors.

### Stage Fine Tuning
Instead of simply train the model to predict new states and hope it will be in the right state, we can apply a discriminator, that has to categorize in what state the predicted gene is. This is a much more sensitive training, compared to a simple regression, yet it can achieve great performance.
