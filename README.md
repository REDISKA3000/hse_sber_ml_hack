# hse_sber_ml_hack
![alt text](https://github.com/REDISKA3000/hse_sber_ml_hack/blob/89ce46407a1cf776a8065baee3d349a451731a80/static/neurokillers_titlepage.jpg)

More than ever before, companies need to study their consumers and potential target audience. Thus, Sber, as a progressive firm, strives to better understand its customers in order to improve their customer experience. That is why, we were challenged with identifying the gender of the client based on the transaction history.

Initially, we received transactions data, where every transaction is described by severall features.
- **client_id** - id of client that completed transaction
- **trans_time** - day from some starting day and time of transaction completion)
- **mcc_code** - merchant category code
- **trans_type** - type of transaction
- **amount** - amount of money that the cilent spent or received
- **term_id** - id of terminal
- **trans_city** - city, where the transaction was completed
- **gender** - our traget feature

![alt text](https://github.com/REDISKA3000/hse_sber_ml_hack/blob/d0a54c852ea16279efea199ace7fed0a859b7960/static/ds_hsesber.jpg)
Then we started from feature engineering appling grouping by client_id and every feature (except for **term_id** and **trans_city**) and diverse discriptive statistics to receive comprehensive information about each customer, you can find it in `transformation.py`.

Obtained new datasets `new_train_big.csv` and `new_test_big.csv`, we experimented with distinct ml-models. With the usage of *Optuna* we optimized hyperparameters for gradient boosting algorthms: *CatBoost*, *XGBoost*, *LightGBM*.
After getting first result we decided to undertake feature selection. Firstly we estimated feature importance using *Shap* package, that estimates the Shepley value for each feature.The value shows, how important this feature is on average for the model.
![alt text](https://github.com/REDISKA3000/hse_sber_ml_hack/blob/2e325a34c593cd75961900f39b39040caeee7050/static/fi_hsesber.jpg)

From this plot we can make trustworthy assumptions about how features affected the training of the model.
It can be said that the mearchant categories best help to identify the gender of the client. 
- mcc_5977 - beauty stores
- mcc_5661 - shoes stores
- mcc_5912 - pharmacies
- mcc_5533 - car spare parts and accessories
- mcc_5541 - car service stations
- mcc_5691 - men's and women's clothing stores
- mcc_5621 - women's clothing

As most Sber customers are Russian and CIS citizens, from general experience we assumed that women are more likely do shoppinng in beauty stores, shoes stores, pharmacies and clothing stores. Whereas men are more likely to spend money on car service and car spare parts.

![alt text](https://github.com/REDISKA3000/hse_sber_ml_hack/blob/375572d82e7755220807f85d420602081394ee21/static/shap_force_plot.png)

Again we the usage of *Shap* we managed to prove our assumption and derive that 0 class stand for womend and 1 class stands for men. We can see on the plot above, that features representing women "move" separating line to the left, 0 class. Thus, we are preety confident that 0 class stands for women.

One more significant thing that I should note is that we also attmepted to construct ANN for binary classification, its architecture one can find in `models.py`. The model is really simple, although it was enough to receive roc_auc score is equal to 0.86. 

Our ultimate solution that resulted in roc_auc = 0.8872, was *CatBoostClassifier* trained on the extended dataset, with the following parameters.
```python
params = {'depth': 6, 'learning_rate': 0.1, 'iterations': 500, 'l2_leaf_reg': 7, 'min_data_in_leaf': 1, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}
```
That is how we became TOP-5 team :)

## our team
[Me](https://github.com/REDISKA3000)<br/>
[Solomon](https://github.com/veidlink)<br/>
[Mikhail](https://github.com/Tehnorobot)<br/>
[Vlad]()
