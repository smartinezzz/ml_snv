# split SAMN15453069

import pandas as pd
import numpy as np

file = pd.read_csv('/data/lab/smart/ml/snvs_pred_gs/SAMN15453069_11nb4fa2ne_all_5c.cnt.matrix.tsv',header='infer',index_col='SNV',sep='\t')



# Define a function
def remove_semicolon(x):
    y = str(x).split(';')[0]
    return y

# Vectorize the function
vectorized_semicolon = np.vectorize(remove_semicolon)

# Apply the function to the 2D array
scread_ref = vectorized_semicolon(file)
scread_ref = pd.DataFrame(scread_ref,columns=file.columns,index=file.index)


# Define a function
def remove_semicolon(x):
    y = str(x).split(';')[1]
    return y

# Vectorize the function
vectorized_semicolon = np.vectorize(remove_semicolon)

# Apply the function to the 2D array
scread_var = vectorized_semicolon(file)
scread_var = pd.DataFrame(scread_var,columns=file.columns,index=file.index)

scread_var.to_csv('/data/lab/smart/ml/snvs_pred_gs/SAMN15453069/SAMN15453069_scread_var_.csv',index=True, sep=',', mode='w')
scread_ref.to_csv('/data/lab/smart/ml/snvs_pred_gs/SAMN15453069/SAMN15453069_scread_ref_.csv',index=True, sep=',', mode='w')


#create gsn file
