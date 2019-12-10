import numpy as np
import torch as t 
import pandas as pd 

d = {'Prob.': [1, 2]}
df = pd.DataFrame(data=d)
df.to_csv("./output.csv",index_label="one",index=False)