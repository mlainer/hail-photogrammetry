This repository contains code for processing, evaluating and analyzing drone-based images of hail for the following publication: 

# Drone-based photogrammetry combined with deep-learning to estimate hail size distributions and melting of hail on the ground

- [Preprint on AMT](https://doi.org/10.5194/amt-2023-89)

Hail is a major threat associated with severe thunderstorms and estimating the hail size is important for issuing warnings to the public. For the validation of existing, operational, radar-derived hail estimates, ground-based observations are necessary. Automatic hail sensors record the kinetic energy of hailstones to estimate the hail sizes. Due to the small size of the observational area of these sensors, the full hail size distribution (HSD) cannot be retrieved. 

Detectron2, a state-of-the-art deep-learning object detection framework (https://github.com/facebookresearch/detectron2) is used to generate a custom-trained model on drone-based aerial photogrammetric data (2D orthomosaic model) to identify hailstones and estimate the size and full hail size distribution (HSD).

## Authors

- [@mlainer](https://www.github.com/mlainer)
