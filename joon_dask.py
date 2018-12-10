import dask.dataframe as dd
import dask.array as da
import h5py
from misc import get_logger, Option

dask_array = []
opt = Option('./config.json')

dataset_dir = '/workspace/dataset/'

# for fn in opt.train_data_list:
#     f = h5py.File('/workspace/dataset/'+fn)
#     print([x for x in f.keys()])
#     d = f['train']['product']
#     print([x for x in d.keys()])
    # array = da.from_array(d, chunks=(opt.chunk_size, ))
    # dask_array.append(array)
# dask_array = da.concatenate(dask_array)
# sr = dd.from_dask_array(dask_array)
# prv = sr.head()
# print(prv)

for fn in opt.train_data_list:
    # f = h5py.File('/workspace/dataset/'+fn)
    # df = dd.read_hdf(f, key='train')

    df = dd.read_hdf(['/workspace/dataset/'+fn], key='/*')

    # df = [dd.read_hdf(f, key=k) for k in f.keys()]

    # print(df.head())

# df = dd.read_hdf('/workspace/dataset/test', key='', chunksize=opt.chunk_size)
# print(df.head())
