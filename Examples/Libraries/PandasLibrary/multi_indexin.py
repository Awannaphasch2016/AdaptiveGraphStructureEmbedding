import numpy as np
import pandas as pd

# index = [('California', 2000), ('California', 2010),
#          ('New York', 2000), ('New York', 2010),
#          ('Texas', 2000), ('Texas', 2010)]
# populations = [33871648, 37253956,
#                18976457, 19378102,
#                20851820, 25145561]
# pop = pd.Series(populations, index=index)
# print(pop)
# index = pd.MultiIndex.from_tuples(index)
# print(index)
#=====================
#==demo
#=====================
demo_index = [('cora', 'run_gcn',200,1),
         ('cora', 'run_gcn', 200, 5),
         ('cora', 'run_gcn', 100, 1),
         ('citeseer', 'run_gcn', 200, 1),
         ('citeseer', 'run_gcn', 200, 5),
         ]
columns = [('class0', 'precision'),
            ('class0', 'f1'),
            ('class1', 'precision'),
            ('class1', 'f1'),
           ]
val = np.arange(20).reshape(5,-1)
col_index = pd.MultiIndex.from_tuples(columns)
row_index = pd.MultiIndex.from_tuples(demo_index)
pop = pd.DataFrame(val, index=demo_index, columns=col_index)
# print(pop)
# print(row_index)
pop = pop.reindex(row_index )
# print(pop)
#--------sort index by level
x = pop.sort_index(axis=0, level=[0,2,3],ascending=False)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(x)


