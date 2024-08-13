from numba import jit
from numba import types
from numba import prange
import numpy as np

###################
# Numba functions #
###################

# Defines the order of regions
ij, ii = range(2)
ijk, iik, iji, ijj, iii = range(5)
ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii = range(15)

@jit(types.Array(types.float64, 1, 'C')(types.Array(types.float64, 2, 'A', readonly=True), types.Array(types.float64, 2, 'A', readonly=True)),nopython=True, cache=True)
def compute_region_means_2_by_2(indicator_0, indicator_1):
    n = indicator_0.shape[0]
    region_sum = np.zeros(15, dtype=np.float64) 
    region_count = np.zeros(15, dtype=np.int64) 
    (ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii) = range(15)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j==k:
                    continue
                region_sum[iikl] += indicator_0[i, i] * indicator_1[j, k]
                region_sum[ijil] += indicator_0[i, j] * indicator_1[i, k]
                region_sum[ijjl] += indicator_0[i, j] * indicator_1[j, k]
                region_sum[ijki] += indicator_0[i, j] * indicator_1[k, i]
                region_sum[ijkj] += indicator_0[i, j] * indicator_1[k, j]
                region_sum[ijkk] += indicator_0[i, j] * indicator_1[k, k]
            region_sum[iiil] += indicator_0[i, i] * indicator_1[i, j]
            region_sum[iiki] += indicator_0[i, i] * indicator_1[j, i]
            region_sum[iikk] += indicator_0[i, i] * indicator_1[j, j]
            region_sum[ijii] += indicator_0[i, j] * indicator_1[i, i]
            region_sum[ijij] += indicator_0[i, j] * indicator_1[i, j]
            region_sum[ijji] += indicator_0[i, j] * indicator_1[j, i]
            region_sum[ijjj] += indicator_0[i, j] * indicator_1[j, j]
        region_sum[iiii] += indicator_0[i, i] * indicator_1[i, i]
    region_count[ijkl] = n*(n-1)*(n-2)*(n-3)
    region_count[iikl] = n*(n-1)*(n-2)
    region_count[ijil] = n*(n-1)*(n-2)
    region_count[ijjl] = n*(n-1)*(n-2)
    region_count[ijki] = n*(n-1)*(n-2)
    region_count[ijkj] = n*(n-1)*(n-2)
    region_count[ijkk] = n*(n-1)*(n-2)
    region_count[iiil] = n*(n-1)
    region_count[iiki] = n*(n-1)
    region_count[iikk] = n*(n-1)
    region_count[ijii] = n*(n-1)
    region_count[ijij] = n*(n-1)
    region_count[ijji] = n*(n-1)
    region_count[ijjj] = n*(n-1)
    region_count[iiii] = n

    region_mean = np.zeros(15, dtype=np.float64)
    for i in range(1,15):
        if region_count[i] > 0:
            region_mean[i] = region_sum[i] / region_count[i]
    if n>3:
        region_mean[ijkl]=indicator_0.mean()*indicator_1.mean()*(n/(n-3))*(n/(n - 2))*(n/(n - 1))-region_sum.sum()/(n*(n-1)*(n-2)*(n-3))
    
    return region_mean

@jit(types.Array(types.float64, 1, 'C')(types.Array(types.float64, 2, 'A', readonly=True), types.Array(types.float64, 2, 'A', readonly=True)),nopython=True, parallel=True, cache=True)
def compute_region_means_2_by_2_parallel(indicator_0, indicator_1):
    n = indicator_0.shape[0]
    region_sum = np.zeros(15, dtype=np.float64) 
    region_count = np.zeros(15, dtype=np.int64) 

    # Temporary arrays for private accumulation
    temp_region_sum = np.zeros((n, 15), dtype=np.float64)
    
    (ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii) = range(15)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j==k:
                    continue
                temp_region_sum[i, iikl] += indicator_0[i, i] * indicator_1[j, k]
                temp_region_sum[i, ijil] += indicator_0[i, j] * indicator_1[i, k]
                temp_region_sum[i, ijjl] += indicator_0[i, j] * indicator_1[j, k]
                temp_region_sum[i, ijki] += indicator_0[i, j] * indicator_1[k, i]
                temp_region_sum[i, ijkj] += indicator_0[i, j] * indicator_1[k, j]
                temp_region_sum[i, ijkk] += indicator_0[i, j] * indicator_1[k, k]
            temp_region_sum[i, iiil] += indicator_0[i, i] * indicator_1[i, j]
            temp_region_sum[i, iiki] += indicator_0[i, i] * indicator_1[j, i]
            temp_region_sum[i, iikk] += indicator_0[i, i] * indicator_1[j, j]
            temp_region_sum[i, ijii] += indicator_0[i, j] * indicator_1[i, i]
            temp_region_sum[i, ijij] += indicator_0[i, j] * indicator_1[i, j]
            temp_region_sum[i, ijji] += indicator_0[i, j] * indicator_1[j, i]
            temp_region_sum[i, ijjj] += indicator_0[i, j] * indicator_1[j, j]
        temp_region_sum[i, iiii] += indicator_0[i, i] * indicator_1[i, i]

    # Reduction step
    for i in range(n):
        region_sum += temp_region_sum[i]
    
    region_count[ijkl] = n*(n-1)*(n-2)*(n-3)
    region_count[iikl] = n*(n-1)*(n-2)
    region_count[ijil] = n*(n-1)*(n-2)
    region_count[ijjl] = n*(n-1)*(n-2)
    region_count[ijki] = n*(n-1)*(n-2)
    region_count[ijkj] = n*(n-1)*(n-2)
    region_count[ijkk] = n*(n-1)*(n-2)
    region_count[iiil] = n*(n-1)
    region_count[iiki] = n*(n-1)
    region_count[iikk] = n*(n-1)
    region_count[ijii] = n*(n-1)
    region_count[ijij] = n*(n-1)
    region_count[ijji] = n*(n-1)
    region_count[ijjj] = n*(n-1)
    region_count[iiii] = n

    region_mean = np.zeros(15, dtype=np.float64)
    for i in range(1,15):
        if region_count[i] > 0:
            region_mean[i] = region_sum[i] / region_count[i]
    if n>3:
        region_mean[ijkl]=indicator_0.mean()*indicator_1.mean()*(n/(n-3))*(n/(n - 2))*(n/(n - 1))-region_sum.sum()/(n*(n-1)*(n-2)*(n-3))
    
    return region_mean

@jit(types.Array(types.float64, 1, 'C')(types.Array(types.float64, 1, 'A', readonly=True), types.Array(types.float64, 2, 'A', readonly=True)),nopython=True, cache=True)
def compute_region_means_1_by_2(indicator_0, indicator_1):
    n = indicator_1.shape[0]
    region_sum = np.zeros(5, dtype=np.float64) 
    region_count = np.zeros(5, dtype=np.int64) 
    (ijk, iik, iji, ijj, iii) = range(5)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            region_sum[iik] += indicator_0[i] * indicator_1[i, j]
            region_sum[iji] += indicator_0[i] * indicator_1[j, i]
            region_sum[ijj] += indicator_0[i] * indicator_1[j, j]
        region_sum[iii] += indicator_0[i] * indicator_1[i, i]
    
    region_count[iik] = n*(n-1)
    region_count[iji] = n*(n-1)
    region_count[ijj] = n*(n-1)
    region_count[iii] = n
    
    region_mean = np.zeros(5, dtype=np.float64)
    for i in range(1,5):
        if region_count[i] > 0:
            region_mean[i] = region_sum[i] / region_count[i]
    if n>2:
        region_mean[ijk]=indicator_0.mean()*indicator_1.mean()*(n/(n - 2))*(n/(n - 1))-region_sum.sum()/(n*(n-1)*(n-2))
    return region_mean


@jit(types.Array(types.float64, 1, 'C')(types.Array(types.float64, 1, 'A', readonly=True), types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True)
def compute_region_means_1_by_1(indicator_0, indicator_1):
    n = indicator_0.shape[0]
    region_sum = np.zeros(2, dtype=np.float64) 
    region_count = np.zeros(2, dtype=np.int64) 
    (ij, ii) = range(2)
    for i in range(n):
        region_sum[ii] += indicator_0[i] * indicator_1[i]
    region_count[ii]=n
    region_mean = np.zeros(2, dtype=np.float64)
    if region_count[ii] > 0:
            region_mean[ii] = region_sum[ii] / region_count[ii]
    if n>1:
        region_mean[ij]=indicator_0.mean()*indicator_1.mean()*(n/(n - 1))-region_sum.sum()/(n*(n-1))
    return region_mean

@jit(types.Array(types.float64, 2, 'C')(types.Array(types.int64, 1, 'A', readonly=True), types.Array(types.float64, 1, 'A', readonly=True)),nopython=True, cache=True)
def mean_inner_product_2_by_2(repetitions,region_mean):
    ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii = range(15)
    n_elements= len(repetitions)
    n=repetitions

    # Create the mean_inner_product array
    mean_inner_product = np.zeros(n_elements**4)
    for i in range(n_elements):
        if n[i] == 0:
            continue
        for j in range(n_elements):
            if i == j or n[j] == 0:
                continue
            for k in range(n_elements):
                if i == k or j==k or n[k] == 0:
                    continue
                for l in range(n_elements):
                    if i == l or j == l or k == l or n[l] == 0:
                        continue 
                    mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + l] = (
                        n[i] * n[j] * n[k] * n[l] * region_mean[ijkl]
                    )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + i * n_elements + k] = (
                    n[i] * n[j] * n[k] * region_mean[ijil] +
                    n[i] * n[j] * n[k] * (n[i] - 1) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + j * n_elements + k] = (
                    n[i] * n[j] * n[k] * region_mean[ijjl] +
                    n[i] * n[j] * n[k] * (n[j] - 1) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + i] = (
                    n[i] * n[j] * n[k] * region_mean[ijki] +
                    n[i] * n[j] * n[k] * (n[i] - 1) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + j] = (
                    n[i] * n[j] * n[k] * region_mean[ijkj] +
                    n[i] * n[j] * n[k] * (n[j] - 1) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + k] = (
                    n[i] * n[j] * n[k] * region_mean[ijkk] +
                    n[i] * n[j] * n[k] * (n[k] - 1) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + i * n_elements**2 + j * n_elements + k] = (
                    n[i] * n[j] * n[k] * region_mean[iikl] +
                    n[i] * n[j] * n[k] * (n[i] - 1) * region_mean[ijkl]
                )
            mean_inner_product[i*n_elements**3+i*n_elements**2+j*n_elements+j] = (
                n[i] * n[j] * region_mean[iikk] +
                n[i] * n[j] * (n[i] - 1) * region_mean[ijkk] +
                n[i] * n[j] * (n[j] - 1) * region_mean[iikl] +
                n[i] * n[j] * (n[i] - 1) * (n[j] - 1) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+j*n_elements**2+i*n_elements+j] = (
                n[i] * n[j] * region_mean[ijij] +
                n[i] * n[j] * (n[i] - 1) * region_mean[ijkj] +
                n[i] * n[j] * (n[j] - 1) * region_mean[ijil] +
                n[i] * n[j] * (n[i] - 1) * (n[j] - 1) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+j*n_elements**2+j*n_elements+i] = (
                n[i] * n[j] * region_mean[ijji] +
                n[i] * n[j] * (n[i] - 1) * region_mean[ijki] +
                n[i] * n[j] * (n[j] - 1) * region_mean[ijjl] +
                n[i] * n[j] * (n[i] - 1) * (n[j] - 1) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+i*n_elements**2+i*n_elements+j] = (
                n[i] * n[j] * region_mean[iiil] +
                n[i] * n[j] * (n[i] - 1) * (region_mean[ijjl] + region_mean[ijil] + region_mean[iikl]) +
                n[i] * n[j] * (n[i] - 1) * (n[i] - 2) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+i*n_elements**2+j*n_elements+i] = (
                n[i] * n[j] * region_mean[iiki] +
                n[i] * n[j] * (n[i] - 1) * (region_mean[ijki] + region_mean[ijkj] + region_mean[iikl]) +
                n[i] * n[j] * (n[i] - 1) * (n[i] - 2) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+j*n_elements**2+j*n_elements+j] = (
                n[i] * n[j] * region_mean[ijjj] +
                n[i] * n[j] * (n[j] - 1) * (region_mean[ijjl] + region_mean[ijkj] + region_mean[ijkk]) +
                n[i] * n[j] * (n[j] - 1) * (n[j] - 2) * region_mean[ijkl]
            )
            mean_inner_product[i*n_elements**3+j*n_elements**2+i*n_elements+i] = (
                n[i] * n[j] * region_mean[ijii] +
                n[i] * n[j] * (n[i] - 1) * (region_mean[ijki] + region_mean[ijil] + region_mean[ijkk]) +
                n[i] * n[j] * (n[i] - 1) * (n[i] - 2) * region_mean[ijkl]
            )
        mean_inner_product[i*n_elements**3+i*n_elements**2+i*n_elements+i] = (
            n[i] * region_mean[iiii] +
            n[i] * (n[i] - 1) * (region_mean[iikk] + region_mean[ijij] + region_mean[ijji] +
            region_mean[iiil] + region_mean[iiki] + region_mean[ijjj] + region_mean[ijii]) +
            n[i] * (n[i] - 1) * (n[i] - 2) * (region_mean[ijil] + region_mean[ijjl] + region_mean[ijki] + region_mean[ijkj] + region_mean[iikl] + region_mean[ijkk]) +
            n[i] * (n[i] - 1) * (n[i] - 2) * (n[i] - 3) * region_mean[ijkl]
        )

    # Flatten the mean_inner array and expand each equation
    return mean_inner_product.reshape(n_elements**2, n_elements**2)

@jit(types.Array(types.float64, 2, 'C')(types.Array(types.int64, 1, 'A', readonly=True), types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True)
def mean_inner_product_1_by_2(repetitions,region_mean):
    ijk, iik, iji, ijj, iii = range(5)
    n_elements= len(repetitions)
    n=repetitions

    # Create the mean_inner_product array
    mean_inner_product = np.zeros(n_elements**3)
    
    for i in range(n_elements):
        for j in range(n_elements):
            for k in range(n_elements):
                id=i*n_elements**2+j*n_elements+k
                if (i == j) and (i == k): #iii
                    mean_inner_product[id]=n[i]*region_mean[iii] + n[i]*(n[i]-1)*(region_mean[iik] + region_mean[iji] + region_mean[ijj])+n[i]*(n[i]-1)*(n[i]-2)*region_mean[ijk]
                elif (j == k) and (i != j): #iij
                    mean_inner_product[id]=n[i]*n[j]*region_mean[ijj] + n[i]*n[j]*(n[j]-1)*region_mean[ijk]
                elif (i == k) and (i != j): #iji
                    mean_inner_product[id]=n[i]*n[j]*region_mean[iji] + n[i]*n[j]*(n[i]-1)*region_mean[ijk]
                elif (i == j) and (i != k): #iik
                    mean_inner_product[id]=n[i]*n[k]*region_mean[iik] + n[i]*n[k]*(n[i]-1)*region_mean[ijk]
                else: #ijk
                    mean_inner_product[id]=n[i]*n[j]*n[k]*region_mean[ijk]
                
    # Flatten the mean_inner array and expand each equation
    return mean_inner_product.reshape(n_elements, n_elements**2)

@jit(types.Array(types.float64, 2, 'C')(types.Array(types.int64, 1, 'A', readonly=True), types.Array(types.float64, 1, 'A', readonly=True)),nopython=True, cache=True)
def mean_inner_product_1_by_1(repetitions,region_mean):
    ij, ii = range(2)
    
    n=repetitions
    n_elements= len(repetitions)

    mean_inner_product = np.zeros(n_elements**2)
    
    for i in range(n_elements):
        for j in range(n_elements):
            id=i*n_elements+j
            if i==j: #ii
                mean_inner_product[id]=n[i]*region_mean[ii]+n[i]*(n[i]-1)*region_mean[ij]
            else: #ij
                mean_inner_product[id]=n[i]*n[j]*region_mean[ij]
 
    return mean_inner_product.reshape(n_elements, n_elements)

@jit(types.Array(types.float64, 2, 'C')(
                 types.Array(types.int64, 1, 'A', readonly=True), 
                 types.Array(types.float64, 2, 'A', readonly=True), 
                 types.Array(types.float64, 3, 'A', readonly=True),
                 types.Array(types.float64, 3, 'A', readonly=True)),
      nopython=True, parallel=True, cache=True)
def build_mean_inner_product_matrix(repetitions, indicators1d, indicators2d, region_means):
    num_matrices1d = len(indicators1d)
    num_matrices2d = len(indicators2d)
    n_elements = len(repetitions)
    num_matrices = num_matrices1d + num_matrices2d
    
    # Compute the size of each block and the total size
    block_sizes = np.empty(num_matrices, dtype=np.int64)
    block_sizes[:num_matrices1d] = n_elements
    block_sizes[num_matrices1d:] = n_elements**2
    total_size = np.sum(block_sizes)
    
    # Create the resulting matrix filled with zeros
    R = np.zeros((total_size, total_size))
        
    # Compute the starting indices for each matrix
    #start_indices = np.cumsum([0] + block_sizes[:-1])
    start_indices=np.zeros(len(block_sizes),dtype=np.int64)
    start=0
    for i in range(1,len(block_sizes)):
        start=start+block_sizes[i-1]
        start_indices[i] = start

    # Create masks for each region
    
    for ij in prange(num_matrices**2):
        i=ij//num_matrices
        j=ij%num_matrices
        if i>j:
            continue
        si, sj = start_indices[i], start_indices[j]
        if i<num_matrices1d and j<num_matrices1d:
            ei, ej = si + n_elements, sj + n_elements
            #region_mean_11 = compute_region_means_1_by_1(indicators1d[i], indicators1d[j])
            R[si:ei, sj:ej] = mean_inner_product_1_by_1(repetitions, region_means[i,j])
        elif i<num_matrices1d and j>=num_matrices1d:
            ei, ej = si + n_elements, sj + n_elements**2
            #region_mean_12 = compute_region_means_1_by_2(indicators1d[i],indicators2d[j-num_matrices1d])
            R[si:ei, sj:ej] = mean_inner_product_1_by_2(repetitions, region_means[i,j])

        elif i>=num_matrices1d and j>=num_matrices1d:
            ei, ej = si + n_elements**2, sj + n_elements**2
            #region_mean_22 = compute_region_means_2_by_2(indicators2d[i-num_matrices1d],indicators2d[j-num_matrices1d])
            R[si:ei, sj:ej] = mean_inner_product_2_by_2(repetitions, region_means[i,j])
        
        if i != j:
            R[sj:ej, si:ei] = R[si:ei, sj:ej].T
            
    return R


@jit(types.Array(types.float64, 3, 'C')(
     types.Array(types.float64, 2, 'A', readonly=True), 
     types.Array(types.float64, 3, 'A', readonly=True)),
      nopython=True, parallel=True, cache=True)
def compute_all_region_means(indicators1d, indicators2d):
    num_matrices1d = len(indicators1d)
    num_matrices2d = len(indicators2d)
    num_matrices = num_matrices1d + num_matrices2d
    
    # Create the resulting matrix filled with zeros
    R = np.zeros((num_matrices,num_matrices,15),dtype=np.float64)
        
    for ij in prange(num_matrices**2):
        i=ij//num_matrices
        j=ij%num_matrices
        if i>j:
            continue
        if i<num_matrices1d and j<num_matrices1d:
            R[i,j,:2] = compute_region_means_1_by_1(indicators1d[i], indicators1d[j])
        elif i<num_matrices1d and j>=num_matrices1d:
            R[i,j,:5] = compute_region_means_1_by_2(indicators1d[i],indicators2d[j-num_matrices1d])
        elif i>=num_matrices1d and j>=num_matrices1d:
            R[i,j,:] = compute_region_means_2_by_2(indicators2d[i-num_matrices1d],indicators2d[j-num_matrices1d])
        R[j,i] = R[i,j]
            
    return R

# #########################
# # Diferential functions #
# #########################

@jit(types.Array(types.float64, 2, 'C')(
     types.int64, 
     types.int64, 
     types.Array(types.int64, 1, 'A', readonly=True),
     types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True)
def diff_mean_inner_product_2_by_2(r0, r1, repetitions, region_mean):
    ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii = range(15)
    n_elements= len(repetitions)
    m=repetitions
    n=m.copy()
    n[r0]-=1
    n[r1]+=1

    # Create the mean_inner_product array
    mean_inner_product = np.zeros(n_elements**4)
    for i in range(n_elements):
        for j in range(n_elements):
            if i == j:
                continue
            for k in range(n_elements):
                if i == k or j==k:
                    continue
                for l in range(n_elements):
                    if i == l or j == l or k == l:
                        continue 
                    mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + l] = (
                        (n[i] * n[j] * n[k] * n[l] - m[i] * m[j] * m[k] * m[l]) * region_mean[ijkl]
                    )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + i * n_elements + k] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[ijil] +
                    (n[i] * n[j] * n[k] * (n[i] - 1) - m[i] * m[j] * m[k] * (m[i] - 1)) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + j * n_elements + k] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[ijjl] +
                    (n[i] * n[j] * n[k] * (n[j] - 1) - m[i] * m[j] * m[k] * (m[j] - 1)) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + i] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[ijki] +
                    (n[i] * n[j] * n[k] * (n[i] - 1) - m[i] * m[j] * m[k] * (m[i] - 1)) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + j] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[ijkj] +
                    (n[i] * n[j] * n[k] * (n[j] - 1) - m[i] * m[j] * m[k] * (m[j] - 1)) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + j * n_elements**2 + k * n_elements + k] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[ijkk] +
                    (n[i] * n[j] * n[k] * (n[k] - 1) - m[i] * m[j] * m[k] * (m[k] - 1)) * region_mean[ijkl]
                )
                mean_inner_product[i * n_elements**3 + i * n_elements**2 + j * n_elements + k] = (
                    (n[i] * n[j] * n[k] - m[i] * m[j] * m[k]) * region_mean[iikl] +
                    (n[i] * n[j] * n[k] * (n[i] - 1) - m[i] * m[j] * m[k] * (m[i] - 1)) * region_mean[ijkl]
                )
            mean_inner_product[i * n_elements**3 + i * n_elements**2 + j * n_elements + j] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[iikk] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * region_mean[ijkk] +
                (n[i] * n[j] * (n[j] - 1) - m[i] * m[j] * (m[j] - 1)) * region_mean[iikl] +
                (n[i] * n[j] * (n[i] - 1) * (n[j] - 1) - m[i] * m[j] * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + j * n_elements**2 + i * n_elements + j] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[ijij] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * region_mean[ijkj] +
                (n[i] * n[j] * (n[j] - 1) - m[i] * m[j] * (m[j] - 1)) * region_mean[ijil] +
                (n[i] * n[j] * (n[i] - 1) * (n[j] - 1) - m[i] * m[j] * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + j * n_elements**2 + j * n_elements + i] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[ijji] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * region_mean[ijki] +
                (n[i] * n[j] * (n[j] - 1) - m[i] * m[j] * (m[j] - 1)) * region_mean[ijjl] +
                (n[i] * n[j] * (n[i] - 1) * (n[j] - 1) - m[i] * m[j] * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + i * n_elements**2 + i * n_elements + j] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[iiil] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * (region_mean[ijjl] + region_mean[ijil] + region_mean[iikl]) +
                (n[i] * n[j] * (n[i] - 1) * (n[i] - 2) - m[i] * m[j] * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + i * n_elements**2 + j * n_elements + i] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[iiki] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * (region_mean[ijki] + region_mean[ijkj] + region_mean[iikl]) +
                (n[i] * n[j] * (n[i] - 1) * (n[i] - 2) - m[i] * m[j] * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + j * n_elements**2 + j * n_elements + j] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[ijjj] +
                (n[i] * n[j] * (n[j] - 1) - m[i] * m[j] * (m[j] - 1)) * (region_mean[ijjl] + region_mean[ijkj] + region_mean[ijkk]) +
                (n[i] * n[j] * (n[j] - 1) * (n[j] - 2) - m[i] * m[j] * (m[j] - 1) * (m[j] - 2)) * region_mean[ijkl]
            )

            mean_inner_product[i * n_elements**3 + j * n_elements**2 + i * n_elements + i] = (
                (n[i] * n[j] - m[i] * m[j]) * region_mean[ijii] +
                (n[i] * n[j] * (n[i] - 1) - m[i] * m[j] * (m[i] - 1)) * (region_mean[ijki] + region_mean[ijil] + region_mean[ijkk]) +
                (n[i] * n[j] * (n[i] - 1) * (n[i] - 2) - m[i] * m[j] * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
            )
        mean_inner_product[i*n_elements**3+i*n_elements**2+i*n_elements+i] = (
            (n[i] - m[i]) * region_mean[iiii] +
            (n[i] * (n[i] - 1) - m[i] * (m[i] - 1)) * (region_mean[iikk] + region_mean[ijij] + region_mean[ijji] + region_mean[iiil] + region_mean[iiki] + region_mean[ijjj] + region_mean[ijii]) +
            (n[i] * (n[i] - 1) * (n[i] - 2) - m[i] * (m[i] - 1) * (m[i] - 2)) * (region_mean[ijil] + region_mean[ijjl] + region_mean[ijki] + region_mean[ijkj] + region_mean[iikl] + region_mean[ijkk]) +
            (n[i] * (n[i] - 1) * (n[i] - 2) * (n[i] - 3) - m[i] * (m[i] - 1) * (m[i] - 2) * (m[i] - 3)) * region_mean[ijkl]
        )

    return mean_inner_product.reshape(n_elements**2, n_elements**2)

@jit(types.Array(types.float64, 2, 'C')(
     types.int64, 
     types.int64, 
     types.Array(types.int64, 1, 'A', readonly=True),
     types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True, fastmath=True)
def diff_mean_inner_product_2_by_2_v2(r0, r1, repetitions, region_mean):
    ijkl, iikl, ijil, ijjl, ijki, ijkj, ijkk, iiil, iiki, iikk, ijii, ijij, ijji, ijjj, iiii = range(15)
    n_elements= len(repetitions)
    m=repetitions
    n=m.copy()
    n[r0]-=1
    n[r1]+=1
    mean_inner_product = np.zeros(n_elements**4)
    n_elements_2=n_elements**2
    n_elements_3=n_elements**3
    # dijkx = np.zeros(n_elements)
    for i in range(n_elements):
        for j in range(i+1,n_elements):
            for k in range(j+1,n_elements):
                nijk=n[i] * n[j] * n[k]
                mijk=m[i] * m[j] * m[k]
                for l in [r0,r1]:
                    if l!=i and l!=j and l!=k:
                        value_ijkl = (nijk * n[l] -mijk * m[l]) * region_mean[ijkl]
                        mean_inner_product[i * n_elements_3 + j * n_elements_2 + k * n_elements + l] = value_ijkl
                        mean_inner_product[i * n_elements_3 + j * n_elements_2 + l * n_elements + k] = value_ijkl
                        mean_inner_product[i * n_elements_3 + k * n_elements_2 + j * n_elements + l] = value_ijkl
                        mean_inner_product[i * n_elements_3 + k * n_elements_2 + l * n_elements + j] = value_ijkl
                        mean_inner_product[i * n_elements_3 + l * n_elements_2 + j * n_elements + k] = value_ijkl
                        mean_inner_product[i * n_elements_3 + l * n_elements_2 + k * n_elements + j] = value_ijkl
                        mean_inner_product[j * n_elements_3 + i * n_elements_2 + k * n_elements + l] = value_ijkl
                        mean_inner_product[j * n_elements_3 + i * n_elements_2 + l * n_elements + k] = value_ijkl
                        mean_inner_product[j * n_elements_3 + k * n_elements_2 + i * n_elements + l] = value_ijkl
                        mean_inner_product[j * n_elements_3 + k * n_elements_2 + l * n_elements + i] = value_ijkl
                        mean_inner_product[j * n_elements_3 + l * n_elements_2 + i * n_elements + k] = value_ijkl
                        mean_inner_product[j * n_elements_3 + l * n_elements_2 + k * n_elements + i] = value_ijkl
                        mean_inner_product[k * n_elements_3 + i * n_elements_2 + j * n_elements + l] = value_ijkl
                        mean_inner_product[k * n_elements_3 + i * n_elements_2 + l * n_elements + j] = value_ijkl
                        mean_inner_product[k * n_elements_3 + j * n_elements_2 + i * n_elements + l] = value_ijkl
                        mean_inner_product[k * n_elements_3 + j * n_elements_2 + l * n_elements + i] = value_ijkl
                        mean_inner_product[k * n_elements_3 + l * n_elements_2 + i * n_elements + j] = value_ijkl
                        mean_inner_product[k * n_elements_3 + l * n_elements_2 + j * n_elements + i] = value_ijkl
                        mean_inner_product[l * n_elements_3 + i * n_elements_2 + j * n_elements + k] = value_ijkl
                        mean_inner_product[l * n_elements_3 + i * n_elements_2 + k * n_elements + j] = value_ijkl
                        mean_inner_product[l * n_elements_3 + j * n_elements_2 + i * n_elements + k] = value_ijkl
                        mean_inner_product[l * n_elements_3 + j * n_elements_2 + k * n_elements + i] = value_ijkl
                        mean_inner_product[l * n_elements_3 + k * n_elements_2 + i * n_elements + j] = value_ijkl
                        mean_inner_product[l * n_elements_3 + k * n_elements_2 + j * n_elements + i] = value_ijkl

                if nijk!=mijk:
                    dijk = nijk - mijk
                    dijki = nijk * (n[i] - 1) - mijk * (m[i] - 1)
                    dijkj = nijk * (n[j] - 1) - mijk * (m[j] - 1)
                    dijkk = nijk * (n[k] - 1) - mijk * (m[k] - 1)
                    dr_ijil = dijk * region_mean[ijil]
                    dr_ijjl = dijk * region_mean[ijjl]
                    dr_ijki = dijk * region_mean[ijki]
                    dr_ijkj = dijk * region_mean[ijkj]
                    dr_ijkk = dijk * region_mean[ijkk]
                    dr_iikl = dijk * region_mean[iikl]
                    dri_ijkl = dijki * region_mean[ijkl]
                    drj_ijkl = dijkj * region_mean[ijkl]
                    drk_ijkl = dijkk * region_mean[ijkl]

                    # i, j, k
                    mean_inner_product[i * n_elements_3 + j * n_elements_2 + i * n_elements + k] = dr_ijil + dri_ijkl
                    mean_inner_product[i * n_elements_3 + j * n_elements_2 + j * n_elements + k] = dr_ijjl + drj_ijkl
                    mean_inner_product[i * n_elements_3 + j * n_elements_2 + k * n_elements + i] = dr_ijki + dri_ijkl
                    mean_inner_product[i * n_elements_3 + j * n_elements_2 + k * n_elements + j] = dr_ijkj + drj_ijkl
                    mean_inner_product[i * n_elements_3 + j * n_elements_2 + k * n_elements + k] = dr_ijkk + drk_ijkl
                    mean_inner_product[i * n_elements_3 + i * n_elements_2 + j * n_elements + k] = dr_iikl + dri_ijkl

                    # i, k, j
                    mean_inner_product[i * n_elements_3 + k * n_elements_2 + i * n_elements + j] = dr_ijil + dri_ijkl
                    mean_inner_product[i * n_elements_3 + k * n_elements_2 + k * n_elements + j] = dr_ijjl + drk_ijkl
                    mean_inner_product[i * n_elements_3 + k * n_elements_2 + j * n_elements + i] = dr_ijki + dri_ijkl
                    mean_inner_product[i * n_elements_3 + k * n_elements_2 + j * n_elements + k] = dr_ijkj + drk_ijkl
                    mean_inner_product[i * n_elements_3 + k * n_elements_2 + j * n_elements + j] = dr_ijkk + drj_ijkl
                    mean_inner_product[i * n_elements_3 + i * n_elements_2 + k * n_elements + j] = dr_iikl + dri_ijkl

                    # j, i, k
                    mean_inner_product[j * n_elements_3 + i * n_elements_2 + j * n_elements + k] = dr_ijil + drj_ijkl
                    mean_inner_product[j * n_elements_3 + i * n_elements_2 + i * n_elements + k] = dr_ijjl + dri_ijkl
                    mean_inner_product[j * n_elements_3 + i * n_elements_2 + k * n_elements + j] = dr_ijki + drj_ijkl
                    mean_inner_product[j * n_elements_3 + i * n_elements_2 + k * n_elements + i] = dr_ijkj + dri_ijkl
                    mean_inner_product[j * n_elements_3 + i * n_elements_2 + k * n_elements + k] = dr_ijkk + drk_ijkl
                    mean_inner_product[j * n_elements_3 + j * n_elements_2 + i * n_elements + k] = dr_iikl + drj_ijkl

                    # j, k, i
                    mean_inner_product[j * n_elements_3 + k * n_elements_2 + j * n_elements + i] = dr_ijil + drj_ijkl
                    mean_inner_product[j * n_elements_3 + k * n_elements_2 + k * n_elements + i] = dr_ijjl + drk_ijkl
                    mean_inner_product[j * n_elements_3 + k * n_elements_2 + i * n_elements + j] = dr_ijki + drj_ijkl
                    mean_inner_product[j * n_elements_3 + k * n_elements_2 + i * n_elements + k] = dr_ijkj + drk_ijkl
                    mean_inner_product[j * n_elements_3 + k * n_elements_2 + i * n_elements + i] = dr_ijkk + dri_ijkl
                    mean_inner_product[j * n_elements_3 + j * n_elements_2 + k * n_elements + i] = dr_iikl + drj_ijkl

                    # k, i, j
                    mean_inner_product[k * n_elements_3 + i * n_elements_2 + k * n_elements + j] = dr_ijil + drk_ijkl
                    mean_inner_product[k * n_elements_3 + i * n_elements_2 + i * n_elements + j] = dr_ijjl + dri_ijkl
                    mean_inner_product[k * n_elements_3 + i * n_elements_2 + j * n_elements + k] = dr_ijki + drk_ijkl
                    mean_inner_product[k * n_elements_3 + i * n_elements_2 + j * n_elements + i] = dr_ijkj + dri_ijkl
                    mean_inner_product[k * n_elements_3 + i * n_elements_2 + j * n_elements + j] = dr_ijkk + drj_ijkl
                    mean_inner_product[k * n_elements_3 + k * n_elements_2 + i * n_elements + j] = dr_iikl + drk_ijkl

                    # k, j, i
                    mean_inner_product[k * n_elements_3 + j * n_elements_2 + k * n_elements + i] = dr_ijil + drk_ijkl
                    mean_inner_product[k * n_elements_3 + j * n_elements_2 + j * n_elements + i] = dr_ijjl + drj_ijkl
                    mean_inner_product[k * n_elements_3 + j * n_elements_2 + i * n_elements + k] = dr_ijki + drk_ijkl
                    mean_inner_product[k * n_elements_3 + j * n_elements_2 + i * n_elements + j] = dr_ijkj + drj_ijkl
                    mean_inner_product[k * n_elements_3 + j * n_elements_2 + i * n_elements + i] = dr_ijkk + dri_ijkl
                    mean_inner_product[k * n_elements_3 + k * n_elements_2 + j * n_elements + i] = dr_iikl + drk_ijkl

        for j in range(n_elements):
            if i==j:
                continue
            nij=n[i] * n[j]
            mij=m[i] * m[j]
            if nij!=mij:
                dij = nij - mij

                mean_inner_product[i * n_elements_3 + i * n_elements_2 + j * n_elements + j] = (
                    (nij - mij) * region_mean[iikk] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * region_mean[ijkk] +
                    (nij * (n[j] - 1) - mij * (m[j] - 1)) * region_mean[iikl] +
                    (nij * (n[i] - 1) * (n[j] - 1) - mij * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + j * n_elements_2 + i * n_elements + j] = (
                    (nij - mij) * region_mean[ijij] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * region_mean[ijkj] +
                    (nij * (n[j] - 1) - mij * (m[j] - 1)) * region_mean[ijil] +
                    (nij * (n[i] - 1) * (n[j] - 1) - mij * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + j * n_elements_2 + j * n_elements + i] = (
                    (nij - mij) * region_mean[ijji] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * region_mean[ijki] +
                    (nij * (n[j] - 1) - mij * (m[j] - 1)) * region_mean[ijjl] +
                    (nij * (n[i] - 1) * (n[j] - 1) - mij * (m[i] - 1) * (m[j] - 1)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + i * n_elements_2 + i * n_elements + j] = (
                    (nij - mij) * region_mean[iiil] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * (region_mean[ijjl] + region_mean[ijil] + region_mean[iikl]) +
                    (nij * (n[i] - 1) * (n[i] - 2) - mij * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + i * n_elements_2 + j * n_elements + i] = (
                    (nij - mij) * region_mean[iiki] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * (region_mean[ijki] + region_mean[ijkj] + region_mean[iikl]) +
                    (nij * (n[i] - 1) * (n[i] - 2) - mij * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + j * n_elements_2 + j * n_elements + j] = (
                    (nij - mij) * region_mean[ijjj] +
                    (nij * (n[j] - 1) - mij * (m[j] - 1)) * (region_mean[ijjl] + region_mean[ijkj] + region_mean[ijkk]) +
                    (nij * (n[j] - 1) * (n[j] - 2) - mij * (m[j] - 1) * (m[j] - 2)) * region_mean[ijkl]
                )

                mean_inner_product[i * n_elements_3 + j * n_elements_2 + i * n_elements + i] = (
                    (nij - mij) * region_mean[ijii] +
                    (nij * (n[i] - 1) - mij * (m[i] - 1)) * (region_mean[ijki] + region_mean[ijil] + region_mean[ijkk]) +
                    (nij * (n[i] - 1) * (n[i] - 2) - mij * (m[i] - 1) * (m[i] - 2)) * region_mean[ijkl]
                )
        if n[i] != m[i]:
            mean_inner_product[i*n_elements_3+i*n_elements_2+i*n_elements+i] = (
                (n[i] - m[i]) * region_mean[iiii] +
                (n[i] * (n[i] - 1) - m[i] * (m[i] - 1)) * (region_mean[iikk] + region_mean[ijij] + region_mean[ijji] + region_mean[iiil] + region_mean[iiki] + region_mean[ijjj] + region_mean[ijii]) +
                (n[i] * (n[i] - 1) * (n[i] - 2) - m[i] * (m[i] - 1) * (m[i] - 2)) * (region_mean[ijil] + region_mean[ijjl] + region_mean[ijki] + region_mean[ijkj] + region_mean[iikl] + region_mean[ijkk]) +
                (n[i] * (n[i] - 1) * (n[i] - 2) * (n[i] - 3) - m[i] * (m[i] - 1) * (m[i] - 2) * (m[i] - 3)) * region_mean[ijkl]
            )

    return mean_inner_product.reshape(n_elements_2, n_elements_2)


@jit(types.Array(types.float64, 2, 'C')(
     types.int64, 
     types.int64, 
     types.Array(types.int64, 1, 'A', readonly=True),
     types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True)
def diff_mean_inner_product_1_by_2(r0, r1, repetitions, region_mean):
    ijk, iik, iji, ijj, iii = range(5)
    n_elements= len(repetitions)
    m=repetitions
    n=m.copy()
    n[r0]-=1
    n[r1]+=1

    # Create the mean_inner_product array
    mean_inner_product = np.zeros(n_elements**3)
    
    for i in range(n_elements):
        for j in range(n_elements):
            for k in range(n_elements):
                id=i*n_elements**2+j*n_elements+k
                if (i == j) and (i == k): #iii
                    mean_inner_product[id]=( (n[i]-m[i])*region_mean[iii] +
                                             (n[i]*(n[i]-1) - m[i]*(m[i]-1))*(region_mean[iik] + region_mean[iji] + region_mean[ijj]) +
                                             (n[i]*(n[i]-1)*(n[i]-2) - m[i]*(m[i]-1)*(m[i]-2))*region_mean[ijk])
                elif (j == k) and (i != j): #iij
                    mean_inner_product[id]=((n[i]*n[j]-m[i]*m[j])*region_mean[ijj] +
                                             (n[i]*n[j]*(n[j]-1)-m[i]*m[j]*(m[j]-1))*region_mean[ijk])
                elif (i == k) and (i != j): #iji
                    mean_inner_product[id]=((n[i]*n[j]-m[i]*m[j])*region_mean[iji] +
                                             (n[i]*n[j]*(n[i]-1)-m[i]*m[j]*(m[i]-1))*region_mean[ijk])
                elif (i == j) and (i != k): #iik
                    mean_inner_product[id]=((n[i]*n[k]-m[i]*m[k])*region_mean[iik] +
                                             (n[i]*n[k]*(n[i]-1)-m[i]*m[k]*(m[i]-1))*region_mean[ijk])
                else: #ijk
                    mean_inner_product[id]=((n[i]*n[j]*n[k]-m[i]*m[j]*m[k])*region_mean[ijk])
                
    # Flatten the mean_inner array and expand each equation
    return mean_inner_product.reshape(n_elements, n_elements**2)

@jit(types.Array(types.float64, 2, 'C')(
     types.int64, 
     types.int64, 
     types.Array(types.int64, 1, 'A', readonly=True),
     types.Array(types.float64, 1, 'A', readonly=True)), nopython=True, cache=True)
def diff_mean_inner_product_1_by_1(r0, r1, repetitions,region_mean):
    ij, ii = range(2)
    
    n_elements= len(repetitions)
    m=repetitions
    n=m.copy()
    n[r0]-=1
    n[r1]+=1
    
    mean_inner_product = np.zeros(n_elements**2)
    
    for i in range(n_elements):
        for j in range(n_elements):
            id=i*n_elements+j
            if i==j: #ii
                mean_inner_product[id]=((n[i]-m[i])*region_mean[ii]+
                                        (n[i]*(n[i]-1)-m[i]*(m[i]-1))*region_mean[ij])
            else: #ij
                mean_inner_product[id]=(n[i]*n[j]-m[i]*m[j])*region_mean[ij]
 
    return mean_inner_product.reshape(n_elements, n_elements)

@jit(types.Array(types.float64, 2, 'C')(
     types.int64, 
     types.int64, 
     types.Array(types.int64, 1, 'A', readonly=True), 
     types.Array(types.float64, 2, 'A', readonly=True), 
     types.Array(types.float64, 3, 'A', readonly=True),
     types.Array(types.float64, 3, 'A', readonly=True)),
     nopython=True, parallel=True, cache=True)
def diff_mean_inner_product_matrix(r0,r1, repetitions, indicators1d, indicators2d, region_means):
    num_matrices1d = len(indicators1d)
    num_matrices2d = len(indicators2d)
    n_elements = len(repetitions)
    num_matrices = num_matrices1d + num_matrices2d
    
    # Compute the size of each block and the total size
    block_sizes = np.empty(num_matrices, dtype=np.int64)
    block_sizes[:num_matrices1d] = n_elements
    block_sizes[num_matrices1d:] = n_elements**2
    total_size = np.sum(block_sizes)
    
    # Create the resulting matrix filled with zeros
    R = np.zeros((total_size, total_size))
        
    # Compute the starting indices for each matrix
    #start_indices = np.cumsum([0] + block_sizes[:-1])
    start_indices=np.zeros(len(block_sizes),dtype=np.int64)
    start=0
    for i in range(1,len(block_sizes)):
        start=start+block_sizes[i-1]
        start_indices[i] = start

    # Create masks for each region
    
    for ij in prange(num_matrices**2):
        i=ij//num_matrices
        j=ij%num_matrices
        if i>j:
            continue
        si, sj = start_indices[i], start_indices[j]
        if i<num_matrices1d and j<num_matrices1d:
            ei, ej = si + n_elements, sj + n_elements
            #region_mean_11 = compute_region_means_1_by_1(indicators1d[i], indicators1d[j])
            R[si:ei, sj:ej] = diff_mean_inner_product_1_by_1(r0,r1,repetitions, region_means[i,j])
        elif i<num_matrices1d and j>=num_matrices1d:
            ei, ej = si + n_elements, sj + n_elements**2
            #region_mean_12 = compute_region_means_1_by_2(indicators1d[i],indicators2d[j-num_matrices1d])
            R[si:ei, sj:ej] = diff_mean_inner_product_1_by_2(r0,r1,repetitions, region_means[i,j])

        elif i>=num_matrices1d and j>=num_matrices1d:
            ei, ej = si + n_elements**2, sj + n_elements**2
            #region_mean_22 = compute_region_means_2_by_2(indicators2d[i-num_matrices1d],indicators2d[j-num_matrices1d])
            R[si:ei, sj:ej] = diff_mean_inner_product_2_by_2(r0,r1,repetitions, region_means[i,j])
        
        if i != j:
            R[sj:ej, si:ei] = R[si:ei, sj:ej].T
            
    return R

###################
# Numpy functions #
###################

def compute_region_means_2_by_2_numpy(indicator_0, indicator_1):
    n=indicator_0.shape[0]
    
    # Outer product of indicators
    indicator_products = np.outer(indicator_0, indicator_1).reshape(n, n, n, n)
    
    # Create arrays of indices
    indices = np.arange(n)
    i, j, k, l = np.meshgrid(indices, indices, indices, indices, indexing='ij', sparse=True)


    masks = {
    # 'ijkl': No index is the same as any other; all four indices are distinct.
    'ijkl': (i != j) & (i != k) & (i != l) & (j != k) & (j != l) & (k != l),
    # 'iikl': The first two indices are equal, and the last two are different from the first two and different from each other.
    'iikl': (i == j) & (i != k) & (i != l) & (k != l),
    # 'ijil, ijjl, ijki, ijkj': Two indices are equal and form a pair, while the other two indices are different from the pair and from each other.
    'ijil': (i != j) & (k != l) & (i == k) & (j != l),
    'ijjl': (i != j) & (k != l) & (j == k) & (i != l),
    'ijki': (i != j) & (k != l) & (i == l) & (j != k),
    'ijkj': (i != j) & (k != l) & (j == l) & (i != k),
    # 'ijkk': The last two indices are equal, and the first two are different from the last two and different from each other.
    'ijkk': (k == l) & (k != i) & (k != j) & (i != j),
    # 'iiil, iiki': The first two indices are equal to one of the last two, the other one of the last two is different.
    'iiil': (i == j) & (i == k) & (i != l),
    'iiki': (i == j) & (i == l) & (i != k),
    # 'iikk': The first two indices form a pair that is equal, and the last two indices form a different pair that is equal, but the two pairs are different from each other.
    'iikk': (i == j) & (k == l) & (i != k),
    # 'ikkk': The last two indices are equal to one of the first two, the other one of the first two is different.
    'ijii': (k == l) & (i == k) & (j != i),
    # 'ijij,ijji': The first and third indices are equal, and the second and fourth indices are equal, but the pairs are different from each other.
    'ijij': (i == k) & (j == l) & (i != j),
    'ijji': (i == l) & (j == k) & (i != j),
    # 'ijjj': The last two indices are equal to one of the first two, the other one of the first two is different.
    'ijjj': (k == l) & (j == k) & (i != j),
    # 'iiii': All indices are equal.
    'iiii': (i == j) & (i == k) & (i == l),
    }
    
    # # Apply masks to the array and store regions in a dictionary
    # region_count = {key: mask.sum() for key, mask in masks.items()}
    # region_mean = {key: (indicator_products[mask].mean() if region_count[key] > 0 else 0) for key, mask in masks.items()}

    # Define the order of regions
    region_order = ['ijkl', 'iikl', 'ijil', 'ijjl', 'ijki', 'ijkj', 'ijkk', 'iiil', 'iiki', 'iikk', 'ijii', 'ijij', 'ijji', 'ijjj', 'iiii']
    
    # Initialize the NumPy array with the correct length
    region_means = np.zeros(len(region_order))
    
    # Calculate region means and store them in the specified order
    for idx, key in enumerate(region_order):
        mask = masks[key]
        region_means[idx] = indicator_products[mask].mean() if mask.any() else 0

    return region_means

def compute_region_means_1_by_2_numpy(indicator_0, indicator_1):
    n = indicator_1.shape[0]
    
    # Outer product of indicators
    indicator_products = np.outer(indicator_0, indicator_1.ravel()).reshape(n, n, n)
    
    # Create arrays of indices
    indices = np.arange(n)
    i, j, k = np.meshgrid(indices, indices, indices, indexing='ij', sparse=True)

    # Define masks for different conditions
    masks = {
        # 'ijk': No index is the same as any other; all three indices are distinct.
        'ijk': (i != j) & (i != k) & (j != k),
        # 'iik': The first two indices are equal, and the last one is different from the first two.
        'iik': (i == j) & (i != k),
        # 'iji': The first and last indices are equal, and the middle one is different from them.
        'iji': (i == k) & (i != j),
        # 'ijj': The last two indices are equal, and the first one is different from them.
        'ijj': (j == k) & (i != j),
        # 'iii': All indices are equal.
        'iii': (i == j) & (i == k)
    }

    # Define the order of regions
    region_order = ['ijk', 'iik', 'iji', 'ijj', 'iii']
    
    # Initialize the NumPy array with the correct length
    region_means = np.zeros(len(region_order))
    
    # Calculate region means and store them in the specified order
    for idx, key in enumerate(region_order):
        mask = masks[key]
        region_means[idx] = indicator_products[mask].mean() if mask.any() else 0

    return region_means

def compute_region_means_1_by_1_numpy(indicator_0, indicator_1):
    n = indicator_0.shape[0]
    
    # Element-wise product of indicators
    indicator_products = np.outer(indicator_0,indicator_1)
    
    # Sum over diagonal elements for 'ii' region
    sum_diagonal = np.sum(np.diag(indicator_products))
    count_diagonal = n
    
     # Create a mask for non-diagonal elements
    mask_non_diagonal = np.ones_like(indicator_products, dtype=bool)
    np.fill_diagonal(mask_non_diagonal, 0)
    
    # Sum over non-diagonal elements for 'ij' region
    sum_non_diagonal = np.sum(indicator_products[mask_non_diagonal])
    count_non_diagonal = n * n - n  # Total elements minus diagonal elements
    
    # Define the order of regions
    #region_order = ['ij', 'ii']
    
    # Initialize the NumPy array with the correct length
    region_means = np.zeros(2)
    
    # Calculate region means and store them in the specified order
    region_means[0] = sum_non_diagonal / count_non_diagonal if count_non_diagonal > 0 else 0
    region_means[1] = sum_diagonal / count_diagonal if count_diagonal > 0 else 0

    return region_means


def compute_region_means(indicator_0,indicator_1,parallel=True, use_numba=True):
    indicator_0=np.array(indicator_0)
    indicator_1=np.array(indicator_1)
    s0=indicator_0.shape[0]
    for d in indicator_0.shape:
        assert d==s0, "All dimensions for the indicators should be the same size"
    for d in indicator_1.shape:
        assert d==s0, "Both indicators should be the same size"
    d0 = len(indicator_0.shape)
    d1 = len(indicator_1.shape)

    if use_numba:
        if d0==d1==1:
            result=compute_region_means_1_by_1(indicator_0, indicator_1)
        elif d0==1 and d1==2:
            result=compute_region_means_1_by_2(indicator_0, indicator_1)
        elif d1==1 and d0==2:
            result=compute_region_means_1_by_2(indicator_1, indicator_0)
        elif d0==d1==2:
            if parallel and d0>100:
                result=compute_region_means_2_by_2(indicator_0, indicator_1)
            else:    
                result=compute_region_means_2_by_2(indicator_0, indicator_1)
        else:
            raise NotImplementedError('This method has not be implemented for dimensions {d0} and {d1}')
        return result
    else:
        if d0==d1==1:
            result=compute_region_means_1_by_1_numpy(indicator_0, indicator_1)
        elif d0==1 and d1==2:
            result=compute_region_means_1_by_2_numpy(indicator_0, indicator_1)
        elif d1==1 and d0==2:
            result=compute_region_means_1_by_2_numpy(indicator_1, indicator_0)
        elif d0==d1==2:
            result=compute_region_means_2_by_2_numpy(indicator_0, indicator_1)
        else:
            raise NotImplementedError('This method has not be implemented for dimensions {d0} and {d1}')
        return result
    
def compute_single_region_means(indicators):
    region_means={}
    for i,i0 in enumerate(indicators):
        for j,i1 in enumerate(indicators):
            result=compute_region_means(i0,i1)
            for key in result:
                region_means[(f'{key}_{i}_{j}')]=result[key]
    return region_means

def mean_inner_product(repetitions,indicator_0,indicator_1, parallel=True, use_numba=True):
    s0=indicator_0.shape[0]
    for d in indicator_0.shape:
        assert d==s0, "All dimensions for the indicators should be the same size"
    for d in indicator_1.shape:
        assert d==s0, "Both indicators should be the same size"
    d0 = len(indicator_0.shape)
    d1 = len(indicator_1.shape)
    
    if d0==d1==1:
        result = mean_inner_product_1_by_1(repetitions,indicator_0, indicator_1)
    elif d0==1 and d1==2:
        result = mean_inner_product_1_by_2(repetitions,indicator_0,indicator_1)
    elif d1==1 and d0==2:
        result = mean_inner_product_1_by_2(repetitions,indicator_1,indicator_0).T
    elif d0==d1==2:
        result = mean_inner_product_2_by_2(repetitions,indicator_0,indicator_1)
    else:
        raise NotImplementedError('This method has not be implemented for indicator dimensions {d0} and {d1}')
    return result 

def combinatorial_numpy_inner_product(native_sequence, indicator_0, indicator_1):
    elements = np.unique(native_sequence)
    len_elements = len(elements)
    len_sequence = len(native_sequence)
    
    # Flatten indicators
    ind_0 = indicator_0.flatten()
    ind_1 = indicator_1.flatten()

    mean_inners = []
    for indicator_0, indicator_1 in [(ind_0, ind_0), (ind_1, ind_0), (ind_0, ind_1), (ind_1, ind_1)]:
        
        # Outer product of indicators
        indicator_products = np.outer(indicator_0, indicator_1).reshape(len_sequence, len_sequence, len_sequence, len_sequence)
        
        # Create arrays of indices
        indices = np.arange(len_sequence)
        i, j, k, l = np.meshgrid(indices, indices, indices, indices, indexing='ij', sparse=True)
        
        def create_region_masks(i,j,k,l):
            masks = {
            # 'ijkl': No index is the same as any other; all four indices are distinct.
            'ijkl': (i != j) & (i != k) & (i != l) & (j != k) & (j != l) & (k != l),
            # 'ijjl': Two indices are equal and form a pair, while the other two indices are different from the pair and from each other.
            'ijjl': (i != j) & (k != l) & (((i == k) & (j != l)) | ((i == l) & (j != k)) | ((j == k) & (i != l)) | ((j == l) & (i != k))),
            # 'iikl': The first two indices are equal, and the last two are different from the first two and different from each other.
            'iikl': (i == j) & (i != k) & (i != l) & (k != l),
            # 'ijkk': The last two indices are equal, and the first two are different from the last two and different from each other.
            'ijkk': (k == l) & (k != i) & (k != j) & (i != j),
            # 'iiil': The first two indices are equal to one of the last two, the other one of the last two is different.
            'iiil': ((i == j) & (i == k) & (i != l)) | ((i == j) & (i == l) & (i != k)),
            # 'ikkk': The last two indices are equal to one of the first two, the other one of the first two is different.
            'ikkk': ((k == l) & (i == k) & (j != i)) | ((k == l) & (j == k) & (i != j)),
            # 'ijij': The first and third indices are equal, and the second and fourth indices are equal, but the pairs are different from each other.
            'ijij': (((i == k) & (j == l)) | ((i == l) & (j == k))) & (i != j),
            # 'iikk': The first two indices form a pair that is equal, and the last two indices form a different pair that is equal, but the two pairs are different from each other.
            'iikk': (i == j) & (k == l) & (i != k),
            # 'iiii': All indices are equal.
            'iiii': (i == j) & (i == k) & (i == l),
            }
            return masks
        
        
        # Define masks
        #Define masks
        masks = create_region_masks(i,j,k,l)

        # Apply masks to the array and store regions in a dictionary
        region_count = {key: mask.sum() for key, mask in masks.items()}
        region_mean = {key: (indicator_products[mask].mean() if region_count[key] > 0 else 0) for key, mask in masks.items()}
        
        # Create the mean_inner_product array
        mean_inner_product = np.zeros((len_elements, len_elements, len_elements, len_elements))
        
        aa_repetitions = np.array([(native_sequence == k).sum() for k in elements])
        
        # Create arrays of indices for elements
        i, j, k, l = np.meshgrid(indices[:len_elements], indices[:len_elements], indices[:len_elements], indices[:len_elements], indexing='ij', sparse=True)
        n_i, n_j, n_k, n_l = np.meshgrid(aa_repetitions, aa_repetitions, aa_repetitions, aa_repetitions, indexing='ij', sparse=False)
        
        # Define masks for elements
        masks = create_region_masks(i,j,k,l)
        
        m = masks['iiii']
        mean_inner_product[m] = (
            n_i[m] * region_mean['iiii'] +
            n_i[m] * (n_i[m]-1) * (region_mean['iikk'] + 2*region_mean['ijij'] + 2*region_mean['iiil'] + 2*region_mean['ikkk']) +
            n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (4*region_mean['ijjl'] + region_mean['iikl'] + region_mean['ijkk']) +
            n_i[m] * (n_i[m]-1) * (n_i[m]-2) * (n_i[m]-3) * region_mean['ijkl']
        )
        
        m = masks['iikk']
        mean_inner_product[m] = (
            n_i[m]*n_k[m] * region_mean['iikk'] +
            n_i[m]*n_k[m] * (n_k[m]-1) * region_mean['iikl'] +
            n_i[m]*n_k[m] * (n_i[m]-1) * region_mean['ijkk'] +
            n_i[m]*(n_i[m]-1)*n_k[m]*(n_k[m]-1) * region_mean['ijkl']
        )
        
        m = masks['ijij']
        mean_inner_product[m] = (
            n_i[m]*n_j[m] * region_mean['ijij'] +
            (n_i[m]*n_j[m]*(n_j[m]-1) + n_j[m]*n_i[m]*(n_i[m]-1)) * region_mean['ijjl'] +
            n_i[m]*(n_i[m]-1)*n_j[m]*(n_j[m]-1) * region_mean['ijkl']
        )
        
        m = masks['iiil'] 
        n_x=n_k*(i!=k) + n_l*(i!=l) # Sometimes is k and sometimes is l the one that is different
        mean_inner_product[m] = (
            n_x[m]*n_i[m]* region_mean['iiil'] +
            n_x[m]*n_i[m]*(n_i[m]-1) * (2*region_mean['ijjl'] + region_mean['iikl']) +
            n_x[m]*n_i[m]*(n_i[m]-1)*(n_i[m]-2) * region_mean['ijkl']
        )
        
        m = masks['ikkk'] # Sometimes is i and sometimes is j the one that is different
        n_x=n_i*(i!=k)+n_j*(j!=k)
        mean_inner_product[m] = (
            n_x[m]*n_k[m] * region_mean['ikkk'] +
            n_x[m]*n_k[m]*(n_k[m]-1) * (2*region_mean['ijjl'] + region_mean['ijkk']) +
            n_x[m]*n_k[m]*(n_k[m]-1)*(n_k[m]-2) * region_mean['ijkl']
        )
        
        m = masks['ijjl']
        n_a=n_j*(i==k)+n_j*(i==l)+n_i*(j==k)+n_i*(j==l) #Different pair from first group
        n_b=n_i*(i==k)+n_i*(i==l)+n_j*(j==k)+n_j*(j==l) #Equal pair from first group
        n_c=n_l*(i==k)+n_k*(i==l)+n_l*(j==k)+n_k*(j==l) #Different pair from second group
        mean_inner_product[m] = (
            n_a[m]*n_b[m]*n_c[m] * region_mean['ijjl'] +
            n_a[m]*n_b[m]*(n_b[m]-1)*n_c[m] * region_mean['ijkl']
        )
        
        m = masks['ijkk']
        mean_inner_product[m] = (
            n_i[m]*n_j[m]*n_k[m] * region_mean['ijkk'] +
            n_i[m]*n_j[m]*n_k[m]*(n_k[m]-1) * region_mean['ijkl']
        )
        
        m = masks['iikl']
        mean_inner_product[m] = (
            n_i[m]*n_k[m]*n_l[m] * region_mean['iikl'] +
            n_i[m]*(n_i[m]-1)*n_k[m]*n_l[m] * region_mean['ijkl']
        )
        
        m = masks['ijkl']
        mean_inner_product[m] = (
            n_i[m]*n_j[m]*n_k[m]*n_l[m] * region_mean['ijkl']
        )
        
        # Flatten the mean_inner array and expand each equation
        mean_inners.append(mean_inner_product.reshape(len_elements**2, len_elements**2))

    return np.concatenate([np.concatenate([mean_inners[0], mean_inners[1]]), np.concatenate([mean_inners[2], mean_inners[3]])], axis=1)

from itertools import permutations

def permutation_numpy_inner_product(native_sequence, indicators):
    elements = np.unique(native_sequence)
    decoy_sequences = np.array(list(permutations(native_sequence)))
    
    # Determine the length of phi_decoy
    phi_len = sum(len(elements) if len(ind.shape) == 1 else len(elements)**2 for ind in indicators)
    
    phi_inner_products = []
    for decoy_sequence in decoy_sequences:
        phi_decoy = np.zeros(phi_len)
        offset = 0
        decoy_pairs = (np.array(np.meshgrid(decoy_sequence, decoy_sequence)) * np.array([1, len(elements)])[:, None, None]).sum(axis=0).ravel()
        for indicator in indicators:
            if len(indicator.shape) == 1:  # 1D indicator
                np.add.at(phi_decoy, decoy_sequence + offset, indicator)
                offset += len(elements)
            elif len(indicator.shape) == 2:  # 2D indicator
                np.add.at(phi_decoy, decoy_pairs + offset, indicator.ravel())
                offset += len(elements) ** 2
        
        phi_inner_products.append(np.outer(phi_decoy, phi_decoy))
    
    phi_inner_products = np.array(phi_inner_products)
    mean_phi_inner = phi_inner_products.mean(axis=0)
    
    return mean_phi_inner

def profile_compilation(n_runs=100,n_elements=20, print_timing=False):
    import time
    matrix1d=np.random.rand(n_elements)
    matrix2d=np.random.rand(n_elements,n_elements)
    repetitions=np.random.randint(0,1000,size=n_elements)
    r0=0
    r1=1
    
    functions_to_benchmark = [
        (compute_region_means_1_by_1, (matrix1d, matrix1d)),
        (compute_region_means_1_by_2, (matrix1d, matrix2d)),
        (compute_region_means_2_by_2, (matrix2d, matrix2d)),
        (compute_region_means_2_by_2_parallel, (matrix2d, matrix2d)),
        (mean_inner_product_1_by_1, (repetitions, compute_region_means_1_by_1(matrix1d, matrix1d))),
        (mean_inner_product_1_by_2, (repetitions, compute_region_means_1_by_2(matrix1d, matrix2d))),
        (mean_inner_product_2_by_2, (repetitions, compute_region_means_2_by_2(matrix2d, matrix2d))),
        (build_mean_inner_product_matrix, (repetitions, np.array([matrix1d,matrix1d,matrix1d]), np.array([matrix2d,matrix2d,matrix2d]),
                                           compute_all_region_means(np.array([matrix1d,matrix1d,matrix1d]), np.array([matrix2d,matrix2d,matrix2d])))),
        (compute_all_region_means, (np.array([matrix1d,matrix1d,matrix1d]), np.array([matrix2d,matrix2d,matrix2d]))),
        (diff_mean_inner_product_1_by_1, (r0, r1, repetitions, compute_region_means_1_by_1(matrix1d, matrix1d))),
        (diff_mean_inner_product_1_by_2, (r0, r1, repetitions, compute_region_means_1_by_2(matrix1d, matrix2d))),
        (diff_mean_inner_product_2_by_2, (r0, r1, repetitions, compute_region_means_2_by_2(matrix2d, matrix2d))),
        (diff_mean_inner_product_2_by_2_v2, (r0, r1, repetitions, compute_region_means_2_by_2(matrix2d, matrix2d))),
        (diff_mean_inner_product_matrix, (r0, r1, repetitions, np.array([matrix1d,matrix1d,matrix1d]), np.array([matrix2d,matrix2d,matrix2d]),
                                          compute_all_region_means(np.array([matrix1d,matrix1d,matrix1d]), np.array([matrix2d,matrix2d,matrix2d])))),
        
    ]

    print("Benchmarking functions:")
    for func, args in functions_to_benchmark:
        # Warm-up run
        func(*args)
        
        # Timed runs
        start_time = time.time()
        for _ in range(n_runs):
            func(*args)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs

        print()
        print(f"{func.__name__}: {avg_time:.6f} seconds")
        print("Compiled signatures:", func.signatures)

        signature = func.signatures[0]
        overload = func.overloads[signature]

        # This is the pipeline we want to look at, @njit = nopython pipeline.
        pipeline = 'nopython'

        if overload.metadata:
            if print_timing:
                # Print the information, it's in the overload.metadata dictionary.
                width = 20
                print("\n\nTimings:\n")
                for name, t in overload.metadata['pipeline_times'][pipeline].items():
                    fmt = (f'{name: <{40}}:'
                        f'{t.init:<{width}.6f}'
                        f'{t.run:<{width}.6f}'
                        f'{t.finalize:<{width}.6f}')
                    print(fmt)
            else:
                
                    total_time = sum([t.init + t.run + t.finalize for t in overload.metadata['pipeline_times'][pipeline].values()])
                    print(f"Total compilation time: {total_time:.6f} seconds")
        else:
            print("No timing information available")

    return functions_to_benchmark

if __name__ == '__main__':
    import pytest
    # Test the test_mean_inner_product function and the test_diff_mean_inner_product function from the tests/test_optimization.py file
    pytest.main(['-v', '../../tests/test_optimization.py::test_mean_inner_product', '../../tests/test_optimization.py::test_diff_mean_inner_product'])
    profile_compilation(n_runs=100, n_elements=20, print_timing=False)

    n_elements=20;r0=1; r1=3; repetitions=np.random.randint(0,1000,size=n_elements);matrix2d_1=np.random.rand(n_elements,n_elements);matrix2d_2=np.random.rand(n_elements,n_elements)
    diff1=diff_mean_inner_product_2_by_2(r0, r1, repetitions, compute_region_means_2_by_2(matrix2d_1, matrix2d_2))
    diff2=diff_mean_inner_product_2_by_2_v2(r0, r1, repetitions, compute_region_means_2_by_2(matrix2d_1, matrix2d_2))

    for i in range(n_elements):
        for j in range(n_elements):
            for k in range(n_elements):
                for l in range(n_elements):
                    assert np.isclose(diff1[i*n_elements+j,k*n_elements+l],diff2[i*n_elements+j,k*n_elements+l]), f"Error at {i},{j},{k},{l}, {diff1[i*n_elements+j,k*n_elements+l]},{diff2[i*n_elements+j,k*n_elements+l]}"
    


    


    
    