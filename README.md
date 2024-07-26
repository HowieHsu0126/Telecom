# 第十九届”挑战杯"全国大学生课外学术科技作品竞赛”揭榜挂帅"专项赛暨中国电信第一届”星海杯"

## 文件说明

- Data.ipynb: 生成数据集
- Model.ipynb：测试传统机器学习模型
- FTTTransformer.ipynb：测试深度学习模型（FTTransformer）

## 实验数据

### 机器学习

| 设置                       | LB    | PB    |
|------------------------------------|-------|-------|
| 40feat + use msisdn + XGBoost      |   -   | 0.8100|
| 40feat + wo use msisdn + soft voting + wo adv + wo pseudo + XGB-100/150/200/500       | 0.7645| 0.8021|
| 40feat + use msisdn + soft voting + wo adv + wo pseudo + XGB-100/150/200/500           | 0.7663| 0.8050|
| 40feat + use msisdn + hard voting + wo adv + wo pseudo + XGB-100/150/200/500           | 0.7662| 0.8050|
| 40feat + use msisdn + stacking + wo adv + wo pseudo+ XGB-100/150/200/500             | 0.6971|   -   |
| 45feat + use msisdn + soft voting + wo adv + wo pseudo + XGB-100/150/200/500          | 0.7753|   TBD |
| 45feat + use msisdn + soft voting + adv + pseudo + XGB-100/150/200/500| 0.8023|   TBD |
| 45feat + use msisdn + soft voting + wo adv + pseudo + XGB-100/150/200/500 | 0.8139|   TBD |
| 45feat + use msisdn + soft voting + wo adv + pseudo + XGB-100/150/200 | 0.8154|   TBD |
| 50feat + use msisdn + soft voting + wo adv + pseudo + XGB-100/150/200 | 0.8175|   TBD |
| 50feat + use msisdn + soft voting + wo adv + pseudo + XGB-100/150/200/500/1000 | 0.8123|   TBD |
| 55feat + use msisdn + soft voting + wo adv + pseudo + XGB-100/150/200 | 0.8158|   TBD |

### 深度学习

| 设置                       | LB    | PB    |
|------------------------------------|-------|-------|
| 40feat + FT-Transformer      |   0.8483   | - |
