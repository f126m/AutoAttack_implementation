# AutoAttackの実装
[AutoAttack](https://arxiv.org/pdf/2003.01690.pdf)の非公式実装です。ただし論文内に詳細な記載のない[FAB Attack](https://arxiv.org/pdf/1907.02044.pdf)と[Square Attack](https://arxiv.org/pdf/1912.00049.pdf)については[公式実装](https://github.com/fra31/auto-attack)を用いています（utilsフォルダ内）。


論文については[こちら](https://dcf-short.hatenablog.com/entry/2022/08/01/014306)で解説しています。


## 使用例
MNISTのデータセットを/dataset/MNISTに設置

学習済みモデルは[mma_training](https://github.com/BorealisAI/mma_training)から引用

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir /datasets/MNIST --model ./trained_models/mnist-Linf-MMA-0.45-sd0/model_best.pt
```

