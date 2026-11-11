# 视觉论文与配图参考索引

本清单区分“参考配图形式”和“采用别人的方法或数据”。论文新增图均用本项目结构、样本和真实权重生成，不粘贴外部论文的图或实验数字。

| 论文/项目 | 论文与代码 | 查看位置及借鉴点 | 本文对应处理 |
|---|---|---|---|
| PlantDoc (CoDS-COMAD 2020) | [论文](https://arxiv.org/pdf/1911.10317) · [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) | 图1的真实图像矩阵，图4的输入与解释图配对 | 用本项目七类训练样本制作图像矩阵；从正式权重计算解释图。没有使用其数据或检测框 |
| PlantWild / MVPDR (ACM MM 2024) | [论文](https://arxiv.org/pdf/2408.03120) · [GitHub](https://github.com/tqwei05/MVPDR) | 图2的数据处理流程，图4的分阶段方法概览 | 绘制自己的叶片身份分组、训练及推理路径。不添加未实现的CLIP、多模态原型分支 |
| Grad-CAM (ICCV 2017) | [论文](https://arxiv.org/abs/1610.02391) · [作者代码](https://github.com/ramprs/grad-cam) | 输入、类别激活与叠加图的对应关系 | 以真实类别分数对layer4求梯度，生成同一输入的两模型对比。只是事后解释，不是病斑标注 |
| Deep Residual Learning (CVPR 2016) | [论文](https://arxiv.org/abs/1512.03385) · [作者代码](https://github.com/KaimingHe/deep-residual-networks) | 网络层级和残差捷径的结构化表达 | 根据实际torchvision ResNet18绘制128像素输入下的张量尺寸与残差块 |
| PlantVillage (Frontiers in Plant Science 2016) | [论文](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full) · [作者数据仓库](https://github.com/spMohanty/PlantVillage-Dataset) | 图1的作物类别样本展示，以及训练/数据表之间的对应 | 使用原作者数据，标明出处；不把公开数据图片称为自行采集 |

## 新增图的证据边界

- 方法图：对应现有代码，不声称网络结构创新；训练增强不出现在确定性推理路径。
- 样本图：每类按文件名排序选择第一张训练图像，提供对应路径及SHA256，不按模型表现挑样本。
- Grad-CAM图：种子2026；选取第一个满足“暗化时增强组纠正基线错误”的案例，以及第一个“模糊时增强组仍错误”的案例。属于有规则选择的示例，不代表全部样本的平均效果。
- 两模型CAM使用相同真实类别作为解释目标；这用到了标签，只用于事后分析，不是部署预测所需信息。
- layer4原生空间分辨率为4×4，热图双线性插值到128×128，再逐图min-max归一化。不同图之间颜色强度不可直接作定量比较。
- 图片来源：PlantVillage作者，数据卡标注CC BY-SA 3.0。新增样本/解释拼图遵守相同署名与共享要求；模型结构与量化图是本项目重绘。
