""" Utility functions for data analysis. """

from multiprocessing import Pool, cpu_count
from functools import update_wrapper
import pandas as pd
import scipy.cluster.hierarchy as sch


def common_values(series1, series2):
    """ Shows the differences, intersections and union of two sets. """
    values1 = set(series1)
    values2 = set(series2)
    intersection = set.intersection(values1, values2)
    no_values2 = values1 - values2
    no_values1 = values2 - values1
    total = set.union(values1, values2)

    print('Intersection: {}'.format(len(intersection)))
    print('Total set 1: {}'.format(len(values1)))
    print('Not in set 2: {}'.format(len(no_values2)))
    print('Total set 2: {}'.format(len(values2)))
    print('Not in set 1: {}'.format(len(no_values1)))
    print('Total: {}'.format(len(total)))


def apply_parallel(grouped, func):
    names, groups = zip(*grouped)
    print(names)

    with Pool(cpu_count()) as p:
        ret_list = p.map(func, groups)

    for name, res in zip(names, ret_list):
        res['group_index'] = name
        res.set_index(['group_index'], append=True, inplace=True)
        res.index = res.index.reorder_levels([2, 0, 1])

    return pd.concat(ret_list)


def cluster_corr(ts):
    """
    Clusters the time series according to their correlation.
    Part of the code was taken from here:
    https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/
    CorrelationMatrixClustering.ipynb
    """
    X = ts.corr().fillna(0).values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    columns = [ts.columns.tolist()[i] for i in list((np.argsort(ind)))]
    clmts = ts.reindex_axis(columns, axis=1)
    return clmts, ind


def decorator(d):
    """
    Make function d a decorator: d wraps a function fn.
    (Thanks to "Desing of Computer Programs" by Peter Norvig)
    """

    def _d(fn):
        return update_wrapper(d(fn), fn)

    update_wrapper(_d, d)
    return _d


@decorator
def pandify(scalar_fun):
    """
    The decorated function applies the scalar function to all values
    and accepts Series and DataFrames.
    Args:
        scalar_fun(function): a scalar function
    Returns:
        df_fun(function): a matrix/vector function for pandas DFs and Series
    """

    def df_fun(df):
        if isinstance(df, pd.Series):
            return df.apply(scalar_fun)
        elif isinstance(df, pd.DataFrame):
            return df.apply(lambda x: x.apply(scalar_fun))
        else:
            return None

    return df_fun
