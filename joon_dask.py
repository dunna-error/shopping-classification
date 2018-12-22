import dask.dataframe as dd
import dask.array as da
import h5py
from misc import get_logger, Option
import pandas as pd

dask_array = []
opt = Option('./config.json')
dataset_dir = '/workspace/dataset/'


dd_list = []
for fn in opt.train_data_list:
    f = h5py.File('/workspace/dataset/'+fn)
    g = f.require_group('train')
    temp = dd.concat([dd.from_array(g[k], chunksize=g[k].shape[0], columns=k)
                      for k in g.keys() if k != 'img_feat'], axis=1)
    dd_list.append(temp)

ddf = dd.concat(dd_list, interleave_partitions=True)
print(ddf.columns)
ddf.product = ddf.product.str.decode('utf-8')
ddf.pid = ddf.pid.str.decode('utf-8')

sr_product = pd.Series([v for k, v in ddf.product.iteritems()])
sr_pid = pd.Series([v for k, v in ddf.pid.iteritems()])
df = pd.concat([sr_pid, sr_product], axis=1)
df.columns = ['pid', 'product']
df.to_pickle(dataset_dir+'df_product.pkl')

# sr_product.to_pickle(dataset_dir+'sr_product.pkl')
# print(sr_product.head())
# da.to_hdf5(dataset_dir+'product.h5', 'product', ddf['product'].values)
# df.to_pickle(dataset_dir+'df_product.pkl')
# print()

# corpus = df.
# print(len(df))
