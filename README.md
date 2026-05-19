## 人工智能课实验代码
### 实验一：无信息搜索
- BFS、DFS、UCS
### 实验二：有信息搜索
- Greedy、A*
### 实验三：对抗搜索
- Minimax、Alpha-Beta Pruning
### 实验四：支持向量机
- SVM
### 实验五：决策树
- 决策树、随机森林
### 实验六：贝叶斯分类器
- 朴素贝叶斯，拉普拉斯修正
### 实验七：强化学习
- 值迭代，策略迭代，Exploratory-MC，Q-Learning
### 实验八：无监督学习
- PCA
### 实验九：多层感知机
- MLP 手写数字识别 和 Fashion 服装识别
### 实验十：LeNet 和 RNN
- LeNet 手写数字识别
- RNN 自回归文本生成


### 附录
#### ipynb->pdf 方法
借助 codex 神力 搓了三个文件

- first_h1_as_title.lua 负责将 ipynb 中的第一个 h1 标题作为 pdf 标题，后续的标题全部上升一级
- pandoc_notebook_header.tex 为将要转换的 pdf 排版格式
- ipynb_to_pdf.py 为转换脚本，将 ipynb 转换为 pdf

注：得提前装好 Jupyter 和 Pandoc。

运行方式
```bash
python ipynb_to_pdf.py exp10/rnn.ipynb
```