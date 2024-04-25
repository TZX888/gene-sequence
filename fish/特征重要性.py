import joblib
import matplotlib.pyplot as plt
model=joblib.load('/home/u2208283040/tzx/LSTM/xgboost/xgboost_Kmer.dat')
print("模型保存成功")
import numpy as np
import xgboost as xgb
feature_names= np.load(file="/home/u2208283040/tzx/LSTM/feature_Kmer_11.npy",allow_pickle=True)
feature_names = list(feature_names)
model.get_booster().feature_names = feature_names
xgb.plot_importance(model,height = .5, 
                        max_num_features=10,
                        show_values = False)
plt.savefig('/home/u2208283040/tzx/LSTM/xgboost_特征2.jpg',bbox_inches = 'tight')