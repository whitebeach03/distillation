import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# CIFAR-10データセットの読み込み
train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# クラスのラベル
class_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# 5x10のグリッドに1つのラベルと5つの画像を縦に表示
rows, cols = 6, 10  # 行数を6に変更
fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))

# 余白の調整用パラメータ
plt.subplots_adjust(wspace=0.2, hspace=0.2)

for class_label in range(10):
    # クラスに対応するインデックスを取得
    class_indices = np.where(np.array(train_dataset.targets) == class_label)[0][:5]

    # ラベルを表示
    axes[0, class_label].text(0.5, 0.15, class_labels[class_label], fontsize=15, ha='center', va='center')
    axes[0, class_label].axis('off')

    # 画像を縦に表示
    for i, idx in enumerate(class_indices, 1):
        img, _ = train_dataset[idx]
        img = img.numpy().transpose((1, 2, 0))
        axes[i, class_label].imshow(img)
        axes[i, class_label].axis('off')

plt.show()
plt.savefig('./cifar.png')
