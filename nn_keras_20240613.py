#siera martinez
#2024/6/13

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
import keras
from keras import layers
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.math import confusion_matrix
from keras import metrics
import csv

score_all = list()
loss_all = list()
dir = '/data/lab/smart/ml/snvs_pred_gs/outs_nn/'
samp_list=['SAMN15453069']#SAMN13012145','SAMN13012146','SAMN13012147','SAMN13012148','SAMN13012149','SAMN13012150','SAMN12799275','SAMN12799274','SAMN12799273','SAMN12799270','SAMN12799269','SAMN12799266','SAMN12799264','SAMN12799263','SAMN15453063','SAMN15453064','SAMN15453069','SAMN12799259','SAMN12799258','SAMN12799257','SAMN15453062','SAMN16086829','SAMN16086830']

stats_comb_dt_gsn = pd.DataFrame()
stats_comb_dt_functiongvs = pd.DataFrame()
stats_comb_gbdt_gsn = pd.DataFrame()
stats_comb_gbdt_functiongvs = pd.DataFrame()

for i in range(len(samp_list)):
    this_samp = samp_list[i]
    print(this_samp)
    scread_ref = pd.read_csv('/data/lab/smart/ml/snvs_pred_gs/'+this_samp+'/'+this_samp+'_scread_ref.csv')
    scread_ref = scread_ref.rename(columns={'Unnamed: 0': 'SNV'})
#    scread_ref.set_index('Unnamed: 0', inplace=True)
    scread_var = pd.read_csv('/data/lab/smart/ml/snvs_pred_gs/'+this_samp+'/'+this_samp+'_scread_var.csv')
#    scread_var = scread_var.rename(columns={'Unnamed: 0': 'SNV'})
#    scread_var.set_index('Unnamed: 0', inplace=True)
#    targets = pd.read_csv('/data/lab/smart/ml/snvs_pred_gs/'+this_samp+'/'+this_samp+'_gsr.csv')
#    targets.set_index('SNV', inplace=True)

    # only for SAMN15453069
    germ = pd.read_csv('/data/lab/backup_28samples/SAMN/tsvs_5c_all/germline_rsID_ESP_noCOS_10c/SAMN15453069_germline_rsID_ESP_noCOS_10c_11nb4fa2ne.tsv',sep='\t')
    som = pd.read_csv('/data/lab/backup_28samples/SAMN/tsvs_5c_all/som_NOrsID_noESP_COS_10c/SAMN15453069_som_NOrsID_noESP_COS_10c_11nb4fa2ne.tsv',sep='\t')
    rna = pd.read_csv('/data/lab/backup_28samples/SAMN/tsvs_5c_all/novel_NOrsID_noESP_noCOS_10c/SAMN15453069_novel_NOrsID_noESP_noCOS_10c_11nb4fa2ne.tsv', sep='\t')

    germ['SNV'] = germ['CHROM'].astype(str)+':'+germ['POS'].astype(str)+germ['REF']+germ['ALT']
    rna['SNV'] = rna['CHROM'].astype(str)+':'+rna['POS'].astype(str)+rna['REF']+rna['ALT']
    som['SNV'] = som['CHROM'].astype(str)+':'+som['POS'].astype(str)+som['REF']+som['ALT']
    
    germ_snvs = pd.DataFrame(germ['SNV'].unique(),columns=['SNV'])
    som_snvs = pd.DataFrame(som['SNV'].unique(),columns=['SNV'])
    rna_snvs = pd.DataFrame(rna['SNV'].unique(),columns=['SNV'])
    
    germ_snvs['gsr'] = 0
    som_snvs['gsr'] = 1
    rna_snvs['gsr'] = 2

    scread_ref = scread_ref.merge(germ_snvs,how='inner',on='SNV')
    scread_ref = scread_ref.merge(som_snvs,how='inner',on='SNV')
    scread_ref = scread_ref.merge(rna_snvs,how='inner',on='SNV')

    targets = scread_ref['gsr']
    targets = pd.DataFrame(targets).rename(columns={'0':'gsr'})
    print(targets['gsr'].unique())
    scread_ref = scread_ref.drop(['gsr'],axis=1)
    print(scread_ref.shape)
    
    #targets.replace('germline', 0, inplace=True) 
    #targets.replace('somatic', 1, inplace=True) 
    #targets.replace('novel', 2, inplace=True)

    #categorize chrom info
    #chromposallele = scread_ref.iloc[:,0]
    #chrom = chromposallele.str.split(':',expand=True).iloc[:,0]
    #chrom.replace('X',23, inplace=True) #change X to num
    #chrom.replace('Y',24, inplace=True) #change Y to num
    #posallele = chromposallele.str.split(':',expand=True).iloc[:,1]
    #pos = posallele.str.split('_',expand=True).iloc[:,0]
    #alleles = posallele.str.split('_',expand=True).iloc[:,1]
    #allele_ref = alleles.str.split('>',expand=True).iloc[:,0]
    #allele_var = alleles.str.split('>',expand=True).iloc[:,1]

    #onehotencode the alleles
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(allele_ref.to_numpy().reshape(-1,1))
    #enc.categories_
    #allele_ref = enc.transform(allele_ref.to_numpy().reshape(-1,1)).toarray()
    #allele_var = enc.transform(allele_var.to_numpy().reshape(-1,1)).toarray()
    
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(pos.to_numpy().reshape(-1,1))
    #pos = enc.transform(pos.to_numpy().reshape(-1,1)).toarray()
    
    #chrom = pd.DataFrame(chrom).astype(int)
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(chrom.to_numpy().reshape(-1,1))
    #chrom = enc.transform(chrom.to_numpy().reshape(-1,1)).toarray()
    
    scread_ref = scread_ref.iloc[:,1:]
    scread_var = scread_var.iloc[:,1:]
    
    #X that includes ref, var, vaf
    nrow,ncol = scread_ref.shape
    
    #cat_features = pd.concat([pd.DataFrame(chrom),pd.DataFrame(pos),pd.DataFrame(allele_ref),pd.DataFrame(allele_var)],axis=1)

    from sklearn import preprocessing
    from matplotlib import pyplot as plt

    X_normalize_ref = preprocessing.normalize(scread_ref, axis=0, norm='l2')
    X_normalize_var = preprocessing.normalize(scread_var, axis=0, norm='l2')
    
    #X_normalize_vaf = pd.concat([pd.DataFrame(X_normalize_vaf),cat_features],axis=1)

    X_normalize_ref=pd.DataFrame(X_normalize_ref)
    X_normalize_ref[X_normalize_ref.isna()]=0
    X_normalize_ref[np.isinf(X_normalize_ref)]=0
    
    #X_normalize_ref = pd.concat([pd.DataFrame(X_normalize_ref),cat_features],axis=1)
    
    nroww,ncoll=X_normalize_ref.shape

    X_normalize_var=pd.DataFrame(X_normalize_var)
    X_normalize_var[X_normalize_var.isna()]=0
    X_normalize_var[np.isinf(X_normalize_var)]=0
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X_rvv = pd.concat([pd.DataFrame(X_normalize_ref),pd.DataFrame(X_normalize_var)],axis=1)
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X_rvv, targets['gsr'],random_state=2)
    #sss=StratifiedShuffleSplit(n_splits=2,train_size=.8,random_state=2)
    #X_train, X_test, y_train, y_test = sss.get_n_splits(X_rvv, targets['germsomnov'])
    print(X_test.shape)
    #oversample data
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    X_train = X_resampled
    y_train = y_resampled
    
    nrow,ncol = X_train.shape
    
    X_train_ref = np.array(X_train)[:,0:int(ncol/2)].reshape(1,int((ncol/2)*nrow))
    X_train_var = np.array(X_train)[:,int(ncol/2):int(ncol)].reshape(1,int((ncol/2)*nrow))
    
    X_rvv_train = np.array([X_train_ref,X_train_var]).reshape(2,nrow,int((ncol/2)))
    nrow,ncol = X_test.shape
    
    X_test_ref = np.array(X_test)[:,0:int(ncol/2)].reshape(1,int((ncol/2)*nrow))
    X_test_var = np.array(X_test)[:,int(ncol/2):int(ncol)].reshape(1,int((ncol/2)*nrow))
    
    X_rvv_test = np.array([X_test_ref,X_test_var]).reshape(2,nrow,int((ncol/2)))

    ## implement keras nn
    # define the model
    # make sure each "image" is (x,y,1)
    
    X_rvv_train = np.transpose(X_rvv_train, (1,0,2))
    X_rvv_test = np.transpose(X_rvv_test, (1,0,2))
    n_sam,x,y = X_rvv_train.shape
    print(X_rvv_test.shape)
    
    X_rvv_train = np.expand_dims(X_rvv_train,-1)
    X_rvv_test = np.expand_dims(X_rvv_test,-1)
    
    num_classes = 3
    input_shape = (x,y,1)
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
#    callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3,min_delta=1)
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(1, 1), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dense(20, activation="relu"),
            layers.Dense(5, activation='relu'),
            #layers.Conv2D(64, kernel_size=(1, 1), activation="relu"),
            #layers.MaxPooling2D(pool_size=(1, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    batch_size = 12
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(np.asarray(X_rvv_train).astype('float32'), y_train, batch_size=batch_size, epochs=epochs)#, callbacks=[callback])
    
    score = model.evaluate(np.asarray(X_rvv_test).astype('float32'), y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    y_pred = model.predict(X_rvv_test)    
    
    y_pred = model.predict(X_rvv_test)
    y_pred = np.argmax (y_pred, axis = 1)
    y_test=np.argmax(y_test, axis=1)
    #Create confusion matrix and normalizes it over predicted (columns)
    
    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)

    sns.heatmap(confusion_mat)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(dir+this_samp+'_nnkeras_'+'1l'+'gsn_confusion_matrix.png')
    plt.close()
    
    score_all.append(score[1])
    loss_all.append(score[0])

score_all = pd.concat([pd.DataFrame(samp_list),pd.DataFrame(score_all)],axis=1)
loss_all = pd.concat([pd.DataFrame(samp_list),pd.DataFrame(loss_all)],axis=1)
pd.DataFrame(score_all).to_csv(dir+'nn_keras_accuracy_gsn_l2k33.csv',header=['SAMN','score'],index=None, sep=',', mode='w')
pd.DataFrame(loss_all).to_csv(dir+'nn_keras_loss_gsn_l2k33.csv',header=['SAMN','log-loss'],index=None, sep=',', mode='w')