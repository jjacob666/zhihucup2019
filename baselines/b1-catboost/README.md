## 智源 - 看山杯 专家发现算法大赛 baseline 0.7


**分数（AUC）**：线下 0.719116，线上 0.701722

**方法**：构造用户、问题特征，构造用户问题交叉特征，5折交叉

**模型**：Catboost

**运行环境**：Ubuntu18，CPU32核，内存125G *(实际内存使用峰值约30%)*，显卡RTX2080Ti 


---

#### 特征说明

**1. 用户特征**

| 特征 | 特征说明 
| :------:| :------: | 
| 'gender', 'freq', 'A1', 'B1', 'C1' ... | 用户原始特征 | 
| 'num_atten_topic', 'num_interest_topic' | 用户关注和感兴趣的topic数 | 
| 'most_interest_topic' | 用户最感兴趣的topic | 
| 'min_interest_values', 'max...', 'std...', 'mean...' | 用户topic兴趣值的统计特征 | 

**2. 问题特征**

| 特征 | 特征说明 
| :------:| :------: | 
| 'num_title_sw', 'num_title_w' | 标题 词计数 | 
| 'num_desc_sw', 'num_desc_w' | 描述 词计数 | 
| 'num_qtopic' | topic计数 | 

**3. 用户问题交叉特征**

| 特征 | 特征说明 
| :------:| :------: | 
| 'num_topic_attent_intersection' | 关注topic交集计数 | 
| 'num_topic_interest_intersection' | 兴趣topic交集计数 | 
| 'min_topic_interest...', 'max...', 'std...', 'mean...' | 交集topic兴趣值统计 | 

---

#### 代码及说明

**1. preprocess**: 数据预处理，包括解析列表，重编码id，pickle保存。

&ensp;&ensp;运行时间 1388s，内存占用峰值 125G * 30%

**2. gen_feat**: 构造特征，特征说明如上述。

&ensp;&ensp;运行时间（32核）1764s，内存占用峰值 125G * 20%

&ensp;&ensp;*(注：这里为了加快运算，所以用了多进程 ，windows上 multiprocessing + jupyter可能有bug，建议linux上跑。)*

**3. baseline**: 模型训练预测。

&ensp;&ensp;运行时间（GPU RTX2080Ti）2848s，内存占用峰值 125G * 12%

