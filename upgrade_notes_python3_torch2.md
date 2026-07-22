# Python 2.7 -> Python 3.10 / PyTorch 2.x 升级说明

本次升级以当前环境为准，已完成代码层面的迁移，当前验证环境为：

- Python: 3.10.20
- PyTorch: 2.13.0+cu126
- torchvision: 0.28.0+cu126

## 1. CK 数据集加载路径改为基于脚本路径

### 修改前
```python
self.data = h5py.File('./data/CK_data.h5', 'r', driver='core')
```

### 修改后
```python
import os

data_path = os.path.join(os.path.dirname(__file__), 'data', 'CK_data.h5')
self.data = h5py.File(data_path, 'r', driver='core')
```

### 修改原因
Python 3 下从不同工作目录运行脚本时，`./data/...` 可能导致文件找不到；改为基于脚本文件路径定位数据文件，更稳定。对应文件：CK.py

---

## 2. Python 2 的 `xrange` 改为 Python 3 的 `range`

### 修改前
```python
for j in xrange(len(test_number)):
    for k in xrange(test_number[j]):
```

### 修改后
```python
for j in range(len(test_number)):
    for k in range(test_number[j]):
```

### 修改原因
`xrange` 是 Python 2 的语法，Python 3 中已不存在，需要替换为 `range`。对应文件：CK.py

---

## 3. 训练脚本中统一改为设备无关的 `device` 写法

### 修改前
```python
use_cuda = torch.cuda.is_available()
...
if use_cuda:
    net.cuda()
```

### 修改后
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = device.type == 'cuda'
...
net.to(device)
```

### 修改原因
PyTorch 2.x 推荐使用统一的设备接口，避免直接依赖 `.cuda()` 这种旧式写法，并且更容易兼容 CPU/GPU。对应文件：mainpro_FER.py、mainpro_CK+.py

---

## 4. 移除 `Variable` 和 `volatile=True`

### 修改前
```python
from torch.autograd import Variable
...
inputs, targets = Variable(inputs), Variable(targets)
inputs, targets = Variable(inputs, volatile=True), Variable(targets)
```

### 修改后
```python
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
```

### 修改原因
`Variable` 在新版 PyTorch 中已不再是推荐用法，`volatile=True` 也已移除；新版代码直接使用 Tensor 并结合 `torch.no_grad()` 完成推理阶段的无梯度计算。对应文件：mainpro_FER.py、mainpro_CK+.py、plot_fer2013_confusion_matrix.py、plot_CK+_confusion_matrix.py、visualize.py

---

## 5. 推理阶段改为 `torch.no_grad()`

### 修改前
```python
inputs, targets = Variable(inputs, volatile=True), Variable(targets)
outputs = net(inputs)
```

### 修改后
```python
with torch.no_grad():
    outputs = net(inputs)
```

### 修改原因
`torch.no_grad()` 是 PyTorch 2.x 的标准写法，用来关闭梯度计算，避免显存占用和不必要的计算。对应文件：mainpro_FER.py、mainpro_CK+.py、plot_fer2013_confusion_matrix.py、plot_CK+_confusion_matrix.py、visualize.py

---

## 6. 将 `.data[0]` / `.data` 的旧取值方式改为 `.item()` / `.sum().item()`

### 修改前
```python
train_loss += loss.data[0]
_, predicted = torch.max(outputs.data, 1)
correct += predicted.eq(targets.data).cpu().sum()
```

### 修改后
```python
train_loss += loss.item()
_, predicted = torch.max(outputs, 1)
correct += predicted.eq(targets).sum().item()
```

### 修改原因
`.data` 访问张量值的写法在新版 PyTorch 中已不推荐，容易引发 autograd 相关问题；`item()` 更安全、明确。对应文件：mainpro_FER.py、mainpro_CK+.py、plot_fer2013_confusion_matrix.py、plot_CK+_confusion_matrix.py

---

## 7. checkpoint 保存和加载改为现代方式

### 修改前
```python
torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))
checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))
```

### 修改后
```python
torch.save(state, os.path.join(path, 'PrivateTest_model.pth'))
checkpoint = torch.load(checkpoint_file, map_location=device)
```

### 修改原因
`.t7` 是旧版 PyTorch 习惯格式，PyTorch 2.x 下建议使用 `.pth` / `.pt`，并且使用 `map_location=device` 避免 CPU/GPU 加载时出现设备不匹配的问题。对应文件：mainpro_FER.py、mainpro_CK+.py、plot_fer2013_confusion_matrix.py、plot_CK+_confusion_matrix.py、visualize.py

---

## 8. 删除模型文件和工具脚本中的旧版 `torch.autograd` 依赖

### 修改前
```python
from torch.autograd import Variable
```

### 修改后
```python
# 不再依赖 Variable
```

### 修改原因
新版 PyTorch 已不需要显式使用 `Variable`，直接使用张量即可。对应文件：models/vgg.py、models/resnet.py、utils.py

---

## 9. `softmax` 的调用方式补齐维度参数

### 修改前
```python
score = F.softmax(outputs_avg)
```

### 修改后
```python
score = F.softmax(outputs_avg, dim=0)
```

### 修改原因
PyTorch 2.x 下显式指定 `dim` 可以避免歧义，保证输出行为一致。对应文件：visualize.py

---

## 10. 兼容性验证结果

已完成以下验证：

1. 运行语法检查：
```bash
python -m py_compile $(find . -name '*.py' | sort)
```
结果：通过，无语法错误。

2. 验证模型前向传播：
```python
from models.vgg import VGG
from models.resnet import ResNet18
x = torch.randn(2, 3, 48, 48)
for model in [VGG('VGG19'), ResNet18()]:
    with torch.no_grad():
        y = model(x)
```
结果：VGG 输出 `(2, 7)`，ResNet 输出 `(2, 7)`，说明模型结构在新版 PyTorch 下可正常运行。

3. 验证依赖库：
- torch 2.13.0+cu126
- torchvision 0.28.0+cu126
- numpy 2.2.6
- h5py 3.16.0
- PIL 12.2.0
- scikit-learn 1.7.2

---

## 结论

本次升级已经把项目从旧版 Python 2.7 / PyTorch 0.2 风格迁移到 Python 3.10 / PyTorch 2.x 的兼容写法，主要修复点集中在：

- Python 2 语法兼容性
- 旧版 PyTorch API 迁移
- 设备管理与无梯度推理
- checkpoint 兼容性
- 数据路径与运行目录相关问题
