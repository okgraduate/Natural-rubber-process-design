# Natural-rubber-process-design
本仓库包含一个用于**天然橡胶初加工工艺设计与反向推荐**的 Python 脚本，通过对实验数据的建模，实现：  - 从三种凝固方式（acid / enzyme / natural）的指标表构建统一数据集   - 计算塑性指标 *plasticity*（作为 6 个指标之一）   - 使用类 NNI 的方式对每种凝固方式做数据增强（约 3000 条样本）   - 训练 BP 神经网络完成 **6 指标 → 凝固方式 (acid / enzyme / natural) 三分类**   - 基于增强后的数据，进行**反向工艺推荐**，给出终点状态样本（dry / package）及完整工艺路径示例  
