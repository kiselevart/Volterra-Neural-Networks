
Code release for [Volterra Neural Networks (VNNs)](https://arxiv.org/abs/1910.09616) and [Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6870).  

Patent Information: [Volterra Neural Network and Method](https://patents.google.com/patent/US20210279519A1/en?q=(siddharth+roheda)&oq=siddharth+roheda)

# Citation
If you use our work please cite and acknowledge:

@inproceedings{roheda2020conquering,<br />
  title={Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},<br />
  volume={34},<br />
  number={07},<br />
  pages={11948--11956},<br />
  year={2020}<br />
}<br />

@article{roheda2019volterra,<br />
  title={Volterra Neural Networks (VNNs)},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  journal={arXiv preprint arXiv:1910.09616},<br />
  year={2019}<br />
}<br />



# Training
Use the unified training entrypoint:

python3 train.py --task <cifar|video> --dataset <dataset> --model <model>

Examples:

- CIFAR-10 orthogonal VNN:
  python3 train.py --task cifar --dataset cifar10 --model vnn_ortho --epochs 50 --batch_size 128 --lr 0.01

- UCF101 fusion higher-order VNN:
  python3 train.py --task video --dataset ucf101 --model vnn_fusion_ho --num_workers 8 --batch_size 8 --lr 1e-4

## Weights & Biases (W&B)
W&B is the default logging backend in this training script.

1) Login once on the machine:

wandb login

2) Run training. Project and run names are auto-generated unless overridden:

python3 train.py --task video --dataset ucf101 --model vnn_fusion_ho

Optional overrides:

- --wandb_project <project_name>
- --wandb_name <run_name>
- --wandb_entity <team_or_user>

W&B init failure behavior is configurable:

- Strict tracking (default): --wandb_on_fail abort
- Continue offline and sync later: --wandb_on_fail offline

## Dataset Paths
Configure dataset roots via environment variables or defaults resolved by mypath.py.
Pre-processing of video frames is done on first run and reused afterward.

Optical flow for fusion models is computed online by default.

################################################## <br />

Pre-Trained Model to be added soon.
