import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

marks = np.array([[70],[20],[30],[99],[40],[16],[80]])
# categories, bins = pd.cut(marks,4,retbins=True,labels=['poor','average','good','excellent'])
# print(categories,bins)

enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
enc.fit(marks)

print(enc.transform(marks))
print(marks)