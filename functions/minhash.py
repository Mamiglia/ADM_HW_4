import pandas as pd
import numpy as np


def shingles_matrix_fit(dataframe,cat_cols,num_cols,num_bins):
    numeric_trans = Pipeline(steps=[
        #the imputer fills Na's with the median value
        ('imputer', SimpleImputer(strategy='median')),
        #divided into 30 bins because above 30 it causes loss of elements
        ('discretizer', KBinsDiscretizer(n_bins=num_bins, strategy='quantile',subsample=None))
    ])
    categoric_trans = Pipeline(steps=[
        #the imputer fills Na's with most frequent element
        ('imputer', SimpleImputer(strategy='most_frequent')),
        #we used a min_frequency that gives us the possibility to unify those values that under a certain
        #threshold, to give us more space to work with
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5e-5))
    ])

    ct = ColumnTransformer([
        ('categorical', categoric_trans, cat_cols),
        ('numerical', numeric_trans, num_cols)
    ])
    shingle_matrix = ct.fit_transform(dataframe).T
    
    #this function will return a shingles matrix and a column transformer that we will apply to our query
    #to have the same transformation, because if we would have just used the the pipeline from scratch
    #the fit_transform would have fit our pipeline to the new data in this way we will just use the already
    #fitted pipeline to our query
    return shingle_matrix, ct

def signature_matrix(shingle_matrix, hashes, N_cols):
    # shingle_matrix = a matrix with unique shingles for every document
    # hashes = the hash functions to apply to our shingles matrix
    # N_cols = number of columns in our matrix
    
    # in the scipy sparse matrix the indices represent an array of the positions in increasing order
    # on each column of the the non zero values, without taking in account the rows.
    nnz = shingle_matrix.indices

    # by reshaping the indices to get everytime a single transaction, we performed K hash functions
    # to mimic pseudo-randomness and a permutation by getting in this way a three-dimensional matrix where 
    # rows: are the positions of the ones in the shingle matrix
    # columns: the transactions
    # depth: the number of hash functions of our minhash
    hash_idx = np.stack([hash(nnz).reshape((-1,N_cols)) for hash in hashes])

    # with matrix.min(axis=2) we go through the third dimension, so the depth
    # and we choose the minimum everytime that will give us the first time a one comes up
    # in every hash function, so we do the over all the depth giving us the signature for every transaction
    signature = hash_idx.min(axis=2)
    
    return signature

def r_hash() :
    '''Returns a random hashing function'''
    a = np.random.randint(1000) + 1 
    b = np.random.randint(1000) 
    m = np.random.randint(1000) * 2 + 1
    return lambda x: (a*x + b) % m

def create_bins(signatures,n_bands):
    
    #initialize a defaultdict that will pass an empty list if the key is not created
    bins = defaultdict(lambda: [])
    
    #iterate through the width of the bands, dropping rows that will not form a complete band
    for band in range(n_bands,len(signatures[:,0])+1,n_bands):
        one_band = signatures[(band - n_bands):band, : ]
        # print(band)
        
        #take a single band of size n_bands and iterate over the columns
        #so that every column of width n_bands will be a key of my dictionary
        for i, row in enumerate(one_band.T):
            bins[str(row)].append(i)
    
    return bins

def minhash(df, cat_cols, num_cols, num_bins=30, K=10, bands_len=2):
    
    #create shingles matrix and pipeline that will be used for our query
    shingle_matrix, pipeline = shingles_matrix_fit(df,cat_cols,num_cols,num_bins)

    #K hashes based on preference
    hashes = [r_hash() for _ in range(K)]

    #tranform our shingles matrix in a signature matrix
    signature = signature_matrix(shingle_matrix, hashes, len(cat_cols) + len(num_cols))

    bins = create_bins(signature, bands_len)

    return signature, bins, pipeline, hashes

def query_minhash(query, bins, pipeline, hashes, bands_len):
    # Returns plausible similar items for every query element
    candidates = defaultdict(lambda: set())

    # get the signature matrix for the query
    query_shingle = pipeline.transform(query)
    query_sig = signature_matrix(query_shingle, hashes, len(query.columns))

    # iterate through bands of our query
    for band in range(bands_len,len(query_sig[:,0])+1,bands_len):
        one_band = query_sig[(band - bands_len):band, : ]

        # save the index of our query and the row of the bands
        for query_num,row in enumerate(one_band.T):

            # if the row is equal to a key in our bins dictionary go with the similarity
            if str(row) in bins:
                cand = bins[str(row)]
                candidates[query_num].update(cand)

    # return a list of lists where every list is the similarity score with a specific query
    return candidates, query_shingle

def jaccard(d1, d2):
    intersect = len(d1.intersection(d2))
    union = len(d1.union(d2))
    return round(float(intersect / union),3)

def query_similarity(row):
    query = set(qs[row.query_number, :].indices)
    candidate = set(sm[:,row.candidate].indices)

    return jaccard(query, candidate)

def most_similar(query, bins, pipeline, hashes, bands_width, threshold=0.7):
    candidates, qs = query_minhash(query, bins, pipeline, hashes, bands_width)

    cand = pd.DataFrame(
        [(q, c) for q, cand in candidates.items() for c in cand ], 
        columns=['query_number', 'candidate'])

    cand['sim'] = cand.apply(query_similarity, axis=1)
    cand = cand.reset_index(drop=True)
    cand = cand[cand.sim > threshold]
    return cand
