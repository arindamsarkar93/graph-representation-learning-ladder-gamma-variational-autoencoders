# Graph Representation Learning via Ladder Gamma Variational Autoencoders

## Abstract
We present a probabilistic framework for community discov-ery and link prediction for graph-structured data, based on a novel, gamma ladder variational autoencoder (VAE) architecture. We model each node in the graph via a deep hierarchy of gamma-distributed embeddings, and define each link probability via a nonlinear function of the bottom-most layer’s embeddings of its associated nodes. In addition to leveraging the representational power of multiple layers of stochastic variables via the ladder VAE architecture, our framework offers the following benefits: 
(1) Unlike existing ladder VAE architectures  based  on  real-valued  latent  variables, the  gamma-distributed latent variables naturally result in non-negativity and sparsity of the learned embeddings, and facilitate their direct interpretation as membership of nodes into (possibly multiple) communities/topics; (2) A novel recognition model for our gamma ladder VAE architecture allows fast inference of node embeddings; and 
(3) The framework also extends nat-urally to incorporate node side information (features and/orlabels). 
Our framework is also fairly modular and can leverage a wide variety of graph neural networks as the VAE encoder. We report both quantitative and qualitative results onseveral benchmark datasets and compare our model with several state-of-the-art methods.

## Example usage for edge-prediction:
python train.py --dataset cora --hidden 64_32 --use_kl_warmup 1 --epochs 500 --cosine_norm 1 --reconstruct_x 1

## Example usage for semisupervised node classification:
python train.py --dataset citeseer --hidden 64_32 --use_kl_warmup 1 --epochs 100  --weight_decay 5e-3 --semisup_train 1 --dropout 0.5 --reconstruct_x 1

## Cite

Please cite our paper if you use this code in your own work:

```
@article{Sarkar_Mehta_Rai_2020, 
  title={Graph Representation Learning via Ladder Gamma Variational Autoencoders}, 
  volume={34}, url={https://ojs.aaai.org/index.php/AAAI/article/view/6013}, DOI={10.1609/aaai.v34i04.6013}, 
  number={04}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Sarkar, Arindam and Mehta, Nikhil and Rai, Piyush}, year={2020}, month={Apr.}, 
  pages={5604-5611} 
 }
```
