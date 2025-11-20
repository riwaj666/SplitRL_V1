SplitRL: Reinforcement Learning for DNN Partitioning
A reinforcement learning framework for learning optimal split points in distributed DNN inference across edge and cloud devices.

Overview
SplitRL uses the REINFORCE algorithm to automatically learn where to partition deep neural networks for edge-cloud collaborative inference, eliminating the need for exhaustive profiling.

Note: This is a research project from CPE-6943. The code is currently in development.

Features
Learns optimal block-level partitioning across diverse DNN architectures
Achieves near-optimal performance (within 1-2 blocks of optimal split)
Generalizes across 6 different models (ResNet, MobileNet, VGG, InceptionV3, AlexNet)
Fast inference (<10ms prediction time)
Results
Pi-to-GPU Configuration
InceptionV3: 3.42 imgs/s (exceeds optimal 3.40 imgs/s)
ResNet50: 1.97 imgs/s (near-optimal)
AlexNet: Near-optimal performance
Pi-to-Pi Configuration
Consistent near-optimal performance across all models
Converges in ~8,000 episodes (3 hours on single GPU)
Methodology
Trained using REINFORCE with:

15,000 episodes across 6 DNN architectures
Learning rate: 1e-3 with exponential decay
Per-model adaptive baselines for variance reduction
Entropy regularization with decay (0.8 → 0.05)
Citation
bibtex
@article{shrestha2024splitrl,
  title={Learning Optimal Split Points: Reinforcement Learning for DNN Partitioning},
  author={Shrestha, Riwaj},
  year={2024},
  institution={University of Texas at San Antonio}
}
Acknowledgments
Dr. Palden Lama (Advisor)
Adiba Masud (Research Support)
Built on profiling data from ParetoPipe framework
License
MIT License

