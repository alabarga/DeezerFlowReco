from scipy.sparse import coo_matrix, csr_matrix
from pyspark.mllib.linalg.distributed import RowMatrix

def compute_similarities(X, sc, threshold=0):
    """ Compute column similarities using Spark:
    Efficient dealing of sparsity with a threshold
    that makes sure that only relevant similarities are computed.
    
    Parameters
    ----------
    X: an array whose features are the rows
    sc: SparkContext
    threshold: the similarity threshold
    
    Return
    ---------
    Symetric similarity matrix shape (X.shape[1], X.shape[1])
    """
    n = X.shape[1]
    rows = sc.parallelize(X)
    mat = RowMatrix(rows)
    
    sims = mat.columnSimilarities(threshold)
    # Convert to scipy sparse matrix
    # Each element is a Matrix entry object (i, j, value)
    rows_index = np.array(sims.entries.map(lambda x: x.i).collect()).astype(int)
    cols_index = np.array(sims.entries.map(lambda x: x.j).collect()).astype(int)
    values = np.array(sims.entries.map(lambda x: x.value).collect())
    triang_sup = coo_matrix((values, (rows_index, cols_index)), shape=(n, n))
    triang_inf = coo_matrix((values, (cols_index, rows_index)), shape=(n, n))
    
    return((triang_sup + triang_inf).tocsr())

#user_features = np.concatenate((train_f_features, user_features), axis=0)
user_features.shape[0] * user_features.shape[1] - np.count_nonzero(user_features)
#More than 20% of the features are equal to 0 -> DIMSUM takes time

from time import time
t0 = time()
user_sim = compute_similarities(user_features, sc, 0.1)
print('Computation took: {0}s'.format(time() - t0))
#sc.stop()
user_sim.nonzero()[0].shape

user_features.shape[1]*10

global_baseline = train_f['is_listened'].mean()
train_f['user_mean'] = train_f['user_id'].map(user_mean_dict).fillna(global_baseline)
train_f['media_mean'] = train_f['media_id'].map(media_mean_dict).fillna(global_baseline)

t0 = time()

k_grid = [5]
for k in k_grid:
    print('k= {0}'.format(k))
    user_position_dict = dict(zip(user_id_list, np.arange(user_id_list.shape[0])))
    position_user_dict = dict(zip(np.arange(user_id_list.shape[0]), user_id_list))

    y_pred = []

    for (current_user, current_media) in test_f[['user_id', 'media_id']].values:
        current_train_f = train_f[train_f['media_id'] == current_media]
        
        # Array of similarities
        current_user_sim = user_sim.getrow(user_position_dict[current_user]).toarray()[0]
        # Only take into account the users who have listened to current_media
        mask_media_id = pd.Series(current_train_f['user_id'].unique()).map(user_position_dict).values
        current_user_sim = current_user_sim[mask_media_id]
        # Top k similarities
        most_corr_users = pd.Series(np.argsort(-current_user_sim)[:k])

        # Array of tuples (user_id, similarity)
        neighbors = zip(pd.Series(mask_media_id[most_corr_users]).map(position_user_dict).values, 
                        current_user_sim[most_corr_users])

        pred = 0
        sim_sum = 0
        for (neighbor_id, sim) in neighbors:
            sim_sum += sim
            df_neighbor = current_train_f[current_train_f['user_id'] == neighbor_id]
            neighbor_mean = df_neighbor['user_mean'].values[0]
            media_mean = df_neighbor['media_mean'].values[0]
            neighbor_is_listened = df_neighbor['is_listened'].values[0]
            neigh_baseline = neighbor_mean + media_mean - global_baseline
            
            pred += sim * (neighbor_is_listened - neigh_baseline)
        
        current_user_baseline = (user_mean_dict.get(current_user, global_baseline)
                                 + media_mean_dict.get(current_media, global_baseline)
                                 - global_baseline)
        if (sim_sum > 0):
            pred = current_user_baseline + (pred / sim_sum)
        else:
            pred = current_user_baseline
            
        y_pred.append(pred)
    
    y_pred = np.array(y_pred)
    y_pred = (y_pred - min(y_pred)) / (max(y_pred) - min(y_pred))
    print('Validation score: {0}'.format(roc_auc_score(test_f['is_listened'], y_pred)))
    print('\n')
    
print('Computation took: {0}s'.format(time() - t0))