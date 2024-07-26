现阶段只需要关注Data.ipynb和Model.ipynb即可。


| Configuration                       | LB    | PB    |
|------------------------------------|-------|-------|
| 40feat + use msisdn + XGBoost      |   -   | 0.8100|
| 40feat + wo use msisdn + soft voting        | 0.7645| 0.8021|
| 40feat + use msisdn + soft voting           | 0.7663| 0.8050|
| 40feat + use msisdn + hard voting           | 0.7662| 0.8050|
| 40feat + use msisdn + stacking              | 0.6971|   -   |
| 45feat + use msisdn + soft voting           | 0.7753|   TBD |
| 45feat + use msisdn + soft voting + adv + pseudo | 0.8023|   TBD |