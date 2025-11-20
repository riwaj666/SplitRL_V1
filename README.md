# SplitRL: Reinforcement Learning for DNN Partitioning

A reinforcement learning framework for learning optimal split points in distributed deep neural network (DNN) inference across edge and cloud devices.

## 🧭 Overview
SplitRL uses the REINFORCE policy-gradient algorithm to automatically learn where to partition DNNs for collaborative edge–cloud execution—removing the need for exhaustive profiling or manual tuning.

Note: This is a research project developed for CPE-6943. The codebase is actively evolving.

## ✨ Features
- Learns optimal block-level partitioning for distributed DNN execution
- Achieves near-optimal performance (within 1–2 blocks of ground truth optimal)
- Generalizes across six architectures: ResNet18/50, MobileNetV2, VGG16, InceptionV3, AlexNet
- Fast split prediction (<10 ms per inference)
- Works for both Pi → GPU and Pi → Pi deployment scenarios

## 📊 Results
### Pi-to-GPU Configuration
- InceptionV3: 3.42 imgs/s (exceeds optimal 3.40 imgs/s)
- ResNet50: 1.97 imgs/s (near-optimal)
- AlexNet: Near-optimal performance

### Pi-to-Pi Configuration
- Consistent near-optimal performance across all models
- Converges in ~8,000 episodes

## 🧠 Methodology
- 15,000 episodes across 6 architectures
- Learning rate: 1e-3 with exponential decay
- Adaptive baselines for variance reduction
- Entropy regularization (0.8 → 0.05)

Reward:
r = -max(p1, p2, network)

## 📚 Citation
@article{shrestha2024splitrl,
  title={Learning Optimal Split Points: Reinforcement Learning for DNN Partitioning},
  author={Shrestha, Riwaj},
  year={2024},
  institution={University of Texas at San Antonio}
}

## 🙏 Acknowledgments
- Dr. Palden Lama
- Adiba Masud
- Based on ParetoPipe profiling framework

## 📜 License
MIT License
