# 房价预测实验说明

## 一、实验目标

本实验要求你使用 `train.csv` 训练一个房价预测模型，并将模型推理代码提交到 Codabench 平台。平台会使用未公开的测试集自动运行你的代码并计算成绩。

预测目标是：

```text
house_price
```

该字段表示房屋总价，单位为“万元”。

## 二、你可以下载的文件

Starting Kit 中包含：

```text
train.csv
predict.py
README.md
```

其中：

- `train.csv`：公开训练数据，包含特征和真实房价。
- `predict.py`：参考提交代码，只用于说明平台输入输出格式，不是最终答案。
- `README.md`：本说明文件。

平台隐藏的 `test.csv` 和真实答案不会提供给学生下载。

## 三、你需要完成的工作

1. 使用 `train.csv` 进行数据分析、特征处理和模型训练。
2. 保存你训练好的模型文件，例如：

```text
model.pkl
model.json
weights.npz
```

3. 修改或重新编写 `predict.py`，让它能够加载你的模型，并对平台隐藏测试集进行预测。
4. 将 `predict.py` 和模型文件一起打包成 zip 文件后提交到 Codabench。

## 四、提交文件要求

你提交的 zip 文件中必须包含：

```text
predict.py
```

如果你的预测代码需要模型文件，也必须一起放入 zip，例如：

```text
predict.py
model.pkl
```

zip 文件本身的名称可以自定义，例如：

```text
team01_submit.zip
final_submission.zip
```

但压缩包内部的主程序必须命名为：

```text
predict.py
```

## 五、平台如何调用你的代码

Codabench 会自动运行：

```bash
python3 predict.py --train train.csv --test test.csv --output predictions.csv
```

参数含义：

- `--train`：公开训练集路径。你的代码可以使用，也可以不使用。
- `--test`：隐藏测试集路径。该文件只有特征，没有 `house_price`。
- `--output`：你需要生成的预测结果文件路径。

如果你已经在本地训练好了模型，`predict.py` 中可以只读取模型文件和 `--test`，不必再次读取 `--train`。

## 六、输出格式要求

你的 `predict.py` 必须生成一个 CSV 文件，列名必须严格为：

```csv
Id,house_price
```

示例：

```csv
Id,house_price
53686,750.0
40585,700.0
```

注意：

- `Id` 必须来自测试集，不能修改。
- `house_price` 必须是数值。
- 预测结果必须覆盖测试集中的所有 `Id`。
- 不要在输出文件中加入多余列。

## 七、评分方式

平台会用隐藏答案计算：

```text
RMSE
MAE
Score
```

排行榜主指标是：

```text
Score
```

Score 越高，排名越高。Score 是一个 0-100 的综合分，由 RMSE 和 MAE 共同计算：

```text
RMSE_score = max(0, 1 - RMSE / baseline_RMSE)
MAE_score  = max(0, 1 - MAE  / baseline_MAE)
Score      = 100 * (0.7 * RMSE_score + 0.3 * MAE_score)
```

其中基线模型是“所有测试样本都预测训练集房价中位数”的简单模型。模型效果越明显优于简单中位数基线，分数越高；效果接近或差于基线时，分数接近 0。

每天最多可提交 10 次，请合理安排提交次数，不要只依赖排行榜反复试错。

## 八、注意事项

- 不允许使用隐藏测试集答案。
- 不要提交 `train.csv`。
- 不要提交 `test.csv`。
- 不要提交 `truth.csv` 或 `solution.csv`。
- 不要共享提交文件给其他团队。
- 如果使用第三方库，请确认平台环境支持；否则建议使用 `pandas`、`numpy`、`scikit-learn` 等常见库。
- 每天提交次数以平台设置为准。

## 九、运行环境要求

本比赛在 Codabench 平台上的运行容器配置为：

```text
Docker image: codalab/codalab-legacy:gpu310
Python: 3.10
```

学生可以使用其他 Python 版本在本地训练模型，但最终提交到平台的 `predict.py` 和模型文件必须能在 Python 3.10 环境下运行。为了减少兼容性问题，建议本地也使用 Python 3.10。

`requirements.txt` 中只列出了推荐安装的依赖库。

推荐本地安装方式：

```bash
conda create -n house-price python=3.10
conda activate house-price
pip install -r requirements.txt
```

如果不使用 conda，也可以：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 包含以下类型的常用库：

- 数据处理：`pandas`、`numpy`、`scipy`
- 机器学习：`scikit-learn`
- 深度学习：`tensorflow`、`torch`
- 可视化：`matplotlib`、`seaborn`
- 进度显示：`tqdm`

Codabench 当前运行环境已经测试可用：

```text
torch==2.3.1+cu121
tensorflow==2.16.1
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
```

Codabench 当前运行环境不支持：

```text
xgboost
lightgbm
catboost
```

平台运行时不保证有 GPU。即使 `torch` 和 `tensorflow` 是 CUDA 版本，也请按 CPU 可运行的方式编写提交代码。

如果使用 `joblib` 或 `pickle` 保存 scikit-learn 模型，请尽量保证训练环境和平台环境中的 scikit-learn 版本接近，否则可能出现模型加载失败。

本实验推荐做法：

1. 本地使用 `train.csv` 训练模型。
2. 使用 `pickle` 保存模型，例如 `model.pkl`。
3. 在 `predict.py` 中加载模型。
4. 对平台传入的隐藏 `test.csv` 生成预测。

例如：

```python
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

最终提交 zip 示例：

```text
predict.py
model.pkl
```
