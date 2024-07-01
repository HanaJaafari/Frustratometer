import prody
import scipy.spatial.distance as sdist
import numpy as np
import typing

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def compute_mask(distance_matrix: np.array,
                 maximum_contact_distance: typing.Union[float, None] = None,
                 minimum_sequence_separation: typing.Union[int, None] = None) -> np.array:
    """
    Computes a 2D Boolean mask from a given distance matrix based on a distance cutoff and a sequence separation cutoff.

    Parameters
    ----------
    distance_matrix : np.array
        A 2D array where the element at index [i, j] represents the spatial distance
        between residues i and j. This matrix is assumed to be symmetric.
    maximum_contact_distance : float, optional
        The maximum distance of a contact. Pairs of residues with distances less than this
        threshold are marked as True in the mask. If None, the spatial distance criterion
        is ignored and all distances are included. Default is None.
    minimum_sequence_separation : int, optional
        A minimum sequence distance threshold. Pairs of residues with sequence indices
        differing by at least this value are marked as True in the mask. If None,
        the sequence distance criterion is ignored. Default is None.

    Returns
    -------
    mask : np.array
        A 2D Boolean array of the same dimensions as `distance_matrix`. Elements of the mask
        are True where the residue pairs meet the specified `distance_cutoff` and
        `sequence_distance_cutoff` criteria.

    Examples
    --------
    >>> import numpy as np
    >>> dm = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    >>> print(compute_mask(dm, distance_cutoff=1.5, sequence_distance_cutoff=1))
    [[False  True False]
     [ True False  True]
     [False  True False]]

    .. todo:: Add chain information for sequence separation
    """
    seq_len = len(distance_matrix)
    mask = np.ones([seq_len, seq_len])
    if minimum_sequence_separation is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance >= minimum_sequence_separation
    if maximum_contact_distance is not None:
        mask *= distance_matrix <= maximum_contact_distance

    return mask.astype(np.bool_)


def compute_native_energy(seq: str,
                          potts_model: dict,
                          mask: np.array,
                          ignore_gap_couplings: bool = False,
                          ignore_gap_fields: bool = False) -> float:
    
    """
    Computes the native energy of a protein sequence based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\sum_i h_i + \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    ignore_couplings_of_gaps : bool, optional
        If True, couplings involving gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.
    ignore_fields_of_gaps : bool, optional
        If True, fields corresponding to gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    energy : float
        The computed energy of the protein sequence based on the Potts model and the interaction mask.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> energy = compute_native_energy(seq, potts_model, mask)
    >>> print(f"Computed energy: {energy:.2f}")

    Notes
    -----
    The energy is computed as the sum of the fields and the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
        
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask 

    gap_indices=[int(i) for i,j in enumerate(seq) if j=="-"]

    if ignore_gap_couplings==True:
        if len(gap_indices)>0:
            j_prime[gap_indices,:]=False
            j_prime[:,gap_indices]=False

    if ignore_gap_fields==True:
        if len(gap_indices)>0:
            h[gap_indices]=False

    energy = h.sum() + j_prime.sum() / 2
    return energy

def compute_fields_energy(seq: str,
                          potts_model: dict,
                          ignore_fields_of_gaps: bool = False) -> float:
    """
    Computes the fields energy of a protein sequence based on a given Potts model.
    
    .. math::
        E = \\sum_i h_i
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    ignore_fields_of_gaps : bool, optional
        If True, fields corresponding to gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    fields_energy : float
        The computed fields energy of the protein sequence based on the Potts model

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> fields_energy = compute_fields_energy(seq, potts_model)
    >>> print(f"Computed fields energy: {fields_energy:.2f}")
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    h = -potts_model['h'][range(seq_len), seq_index]
    
    if ignore_fields_of_gaps==True:
        gap_indices=[int(i) for i,j in enumerate(seq) if j=="-"]
        if len(gap_indices)>0:
            h[gap_indices]=False
    fields_energy=h.sum()
    return fields_energy

def compute_couplings_energy(seq: str,
                      potts_model: dict,
                      mask: np.array,
                      ignore_couplings_of_gaps: bool = False) -> float:
    """
    Computes the couplings energy of a protein sequence based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seq : str
        The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequence and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
    ignore_couplings_of_gaps : bool, optional
        If True, couplings involving gaps ('-') in the sequence are set to 0 in the energy calculation.
        Default is False.

    Returns
    -------
    couplings_energy : float
        The computed couplings energy of the protein sequence based on the Potts model and the interaction mask.

    Examples
    --------
    >>> seq = "ACDEFGHIKLMNPQRSTVWY"
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq), len(seq)), dtype=bool) # Include all pairs
    >>> couplings_energy = compute_couplings_energy(seq, potts_model, mask)
    >>> print(f"Computed couplings energy: {couplings_energy:.2f}")

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask
    if ignore_couplings_of_gaps==True:
        gap_indices=[i for i,j in enumerate(seq) if j=="-"]
        if len(gap_indices)>0:
            j_prime[:,gap_indices]=False
            j_prime[gap_indices,:]=False
    couplings_energy = j_prime.sum() / 2
    return couplings_energy

def compute_sequences_energy(seqs: list,
                             potts_model: dict,
                             mask: np.array,
                             split_couplings_and_fields = False) -> np.array:
    """
    Computes the energy of multiple protein sequences based on a given Potts model and an interaction mask.
    
    .. math::
        E = \\sum_i h_i + \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
        
    Parameters
    ----------
    seqs : list
        List of amino acid sequences in string format, separated by commas. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of each sequence (L) should all match the dimensions of the Potts model.
    potts_model : dict
        A dictionary containing the Potts model parameters 'h' (fields) and 'J' (couplings). The fields are a 2D array of shape (L, 20), where L is the length of the sequences and 20 is the number of amino acids. The couplings are a 4D array of shape (L, L, 20, 20). The fields and couplings are assumed to be in units of energy.
    mask : np.array
        A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequences.
    split_couplings_and_fields : bool, optional
        If True, two lists of the sequences' couplings and fields energies are returned.
        Default is False.

    Returns
    -------
    energy (if split_couplings_and_fields==False): float
        The computed energies of the protein sequences based on the Potts model and the interaction mask.
    fields_couplings_energy (if split_couplings_and_fields==True): np.array
        Array containing computed fields and couplings energies of the protein sequences based on the Potts model and the interaction mask. 

    Examples
    --------
    >>> seq_list = ["ACDEFGHIKLMNPQRSTVWY","AKLWYMNPQRSTCDEFGHIV"]
    >>> potts_model = {
        'h': np.random.rand(20, 20),  # Random fields
        'J': np.random.rand(20, 20, 20, 20)  # Random couplings
    }
    >>> mask = np.ones((len(seq_list[0]), len(seq_list[0])), dtype=bool) # Include all pairs
    >>> energies = compute_sequences_energy(seq_list, potts_model, mask)
    >>> print(f"Sequence 1 energy: {energies[0]:.2f}")
    >>> print(f"Sequence 2 energy: {energies[1]:.2f}")

    Notes
    -----
    The couplings energy is computed as the half-sum of the couplings for all pairs of residues
    where the mask is True. The division by 2 for the couplings accounts for double-counting in symmetric
    matrices.

    .. todo:: Optimize the computation.
    """

    seq_index = np.array([[_AA.find(aa) for aa in seq] for seq in seqs])
    N_seqs, seq_len = seq_index.shape
    pos_index=np.repeat([np.arange(seq_len)], N_seqs,axis=0)


    pos1=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[0] for p in pos_index])
    pos2=np.array([np.meshgrid(p, p, indexing='ij', sparse=True)[1] for p in pos_index])
    aa1=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[0] for s in seq_index])
    aa2=np.array([np.meshgrid(s, s, indexing='ij', sparse=True)[1] for s in seq_index])
    
    h = -potts_model['h'][pos_index,seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * mask

    if split_couplings_and_fields:
        fields_couplings_energy=np.array([h.sum(axis=-1),j_prime.sum(axis=-1).sum(axis=-1) / 2])
        return fields_couplings_energy
    else:
        energy = h.sum(axis=-1) + j_prime.sum(axis=-1).sum(axis=-1) / 2
        return energy


def compute_singleresidue_decoy_energy_fluctuation(seq: str,
                                                   potts_model: dict,
                                                   mask: np.array) -> np.array:
    r"""
    $ \Delta H_i = \Delta h_i + \sum_k\Delta j_{ik} $

    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, aa1 = np.meshgrid(np.arange(seq_len), np.arange(21), indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1

    j_correction = np.zeros([seq_len, seq_len, 21])
    # J correction interactions with other aminoacids
    reduced_j = potts_model['J'][range(seq_len), :, seq_index, :].astype(np.float32)
    j_correction += reduced_j[:, pos1, seq_index[pos1]] * mask[:, pos1]
    j_correction -= reduced_j[:, pos1, aa1] * mask[:, pos1]

    # J correction, interaction with self aminoacids
    decoy_energy += j_correction.sum(axis=0)

    return decoy_energy


def compute_mutational_decoy_energy_fluctuation(seq: str,
                                                potts_model: dict,
                                                mask: np.array, ) -> np.array:
    r"""
    $$ \Delta DCA_{ij} = H_i - H_{i'} + H_{j}-H_{j'}
    + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j}
    + \sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
    $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create masked decoys
    pos1,pos2=np.where(mask>0)
    contacts_len=len(pos1)

    pos1,aa1,aa2=np.meshgrid(pos1, np.arange(21), np.arange(21), indexing='ij', sparse=True)
    pos2,aa1,aa2=np.meshgrid(pos2, np.arange(21), np.arange(21), indexing='ij', sparse=True)

    #Compute fields
    decoy_energy = np.zeros([contacts_len, 21, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    decoy_energy -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2

    #Compute couplings
    j_correction = np.zeros([contacts_len, 21, 21])
    for pos, aa in enumerate(seq_index):
        # J correction interactions with other aminoacids
        reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
        j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= reduced_j[pos1, aa1] * mask[pos, pos1]
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask[pos, pos2]
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    decoy_energy += j_correction
    
    decoy_energy2=np.zeros([seq_len,seq_len,21,21])
    decoy_energy2[mask]=decoy_energy
    return decoy_energy2


def compute_configurational_decoy_energy_fluctuation(seq: str,
                                                     potts_model: dict,
                                                     mask: np.array, ) -> np.array:
    r"""
    $$ \Delta DCA_{ij} = H_i - H_{i'} + H_{j}-H_{j'}
    + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j}
    + \sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
    $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create masked decoys
    pos1,pos2=np.where(mask>0)
    contacts_len=len(pos1)

    pos1,aa1,aa2=np.meshgrid(pos1, np.arange(21), np.arange(21), indexing='ij', sparse=True)
    pos2,aa1,aa2=np.meshgrid(pos2, np.arange(21), np.arange(21), indexing='ij', sparse=True)

    #Compute fields
    decoy_energy = np.zeros([contacts_len, 21, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    decoy_energy -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2

    #Compute couplings
    j_correction = np.zeros([contacts_len, 21, 21])
    for pos, aa in enumerate(seq_index):
        # J correction interactions with other aminoacids
        reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
        j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= reduced_j[pos1, aa1] * mask.mean()
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask.mean()
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask.mean()  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask.mean()  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask.mean()  # Correct combination
    decoy_energy += j_correction
    
    decoy_energy2=np.zeros([seq_len,seq_len,21,21])
    decoy_energy2[mask]=decoy_energy
    return decoy_energy2


def compute_contact_decoy_energy_fluctuation(seq: str,
                                             potts_model: dict,
                                             mask: np.array) -> np.array:
    r"""
    $$ \Delta DCA_{ij} = \Delta j_{ij} $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """

    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, pos2, aa1, aa2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), np.arange(21), np.arange(21),
                                       indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, seq_len, 21, 21])
    decoy_energy += potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Old coupling
    decoy_energy -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # New Coupling

    return decoy_energy


def compute_decoy_energy(seq: str, potts_model: dict, mask: np.array, kind='singleresidue'):
    """
    Calculates the decoy energy (Obsolete)
    :param seq:
    :param potts_model:
    :param mask:
    :param kind:
    :return:
    """

    native_energy = compute_native_energy(seq, potts_model, mask)
    if kind == 'singleresidue':
        return native_energy + compute_singleresidue_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'mutational':
        return native_energy + compute_mutational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'configurational':
        return native_energy + compute_configurational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'contact':
        return native_energy + compute_contact_decoy_energy_fluctuation(seq, potts_model, mask)


def compute_aa_freq(sequence, include_gaps=True):
    seq_index = np.array([_AA.find(aa) for aa in sequence])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)])
    if not include_gaps:
        aa_freq[0] = 0
    return aa_freq


def compute_contact_freq(sequence):
    seq_index = np.array([_AA.find(aa) for aa in sequence])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)], dtype=np.float64)
    aa_freq /= aa_freq.sum()
    contact_freq = (aa_freq[:, np.newaxis] * aa_freq[np.newaxis, :])
    return contact_freq


def compute_single_frustration(decoy_fluctuation,
                               aa_freq=None,
                               correction=0):
    if aa_freq is None:
        aa_freq = np.ones(21)
    mean_energy = (aa_freq * decoy_fluctuation).sum(axis=1) / aa_freq.sum()
    std_energy = np.sqrt(
        ((aa_freq * (decoy_fluctuation - mean_energy[:, np.newaxis]) ** 2) / aa_freq.sum()).sum(axis=1))
    frustration = -mean_energy / (std_energy + correction)
    return -frustration


def compute_pair_frustration(decoy_fluctuation,
                             contact_freq: typing.Union[None, np.array],
                             correction=0) -> np.array:
    if contact_freq is None:
        contact_freq = np.ones([21, 21])
    decoy_energy = decoy_fluctuation
    seq_len = decoy_fluctuation.shape[0]
    average = np.average(decoy_energy.reshape(seq_len * seq_len, 21 * 21), weights=contact_freq.flatten(), axis=-1)
    variance = np.average((decoy_energy.reshape(seq_len * seq_len, 21 * 21) - average[:, np.newaxis]) ** 2,
                          weights=contact_freq.flatten(), axis=-1)
    mean_energy = average.reshape(seq_len, seq_len)
    std_energy = np.sqrt(variance).reshape(seq_len, seq_len)
    contact_frustration = -mean_energy / (std_energy + correction)
    return -contact_frustration


def compute_scores(potts_model: dict) -> np.array:
    """
    Computes contact scores based on the Frobenius norm

    CN[i,j] = F[i,j] - F[i,:] * F[:,j] / F[:,:]

    Parameters
    ----------
    potts_model :  dict
        Potts model containing the couplings in the "J" key

    Returns
    -------
    scores : np.array
        Score matrix (N x N)
    """
    j = potts_model['J']
    n, _, __, q = j.shape
    norm = np.linalg.norm(j.reshape(n * n, q * q), axis=1).reshape(n, n)  # Frobenius norm
    norm_mean = np.mean(norm, axis=0) / (n - 1) * n
    norm_mean_all = np.mean(norm) / (n - 1) * n
    corr_norm = norm - norm_mean[:, np.newaxis] * norm_mean[np.newaxis, :] / norm_mean_all
    corr_norm[np.diag_indices(n)] = 0
    corr_norm = np.mean([corr_norm, corr_norm.T], axis=0)  # Symmetrize matrix
    return corr_norm


def compute_roc(scores, distance_matrix, cutoff):
    scores = sdist.squareform(scores)
    distance = sdist.squareform(distance_matrix)
    results = np.array([np.array(scores), np.array(distance)])
    results = results[:, results[0, :].argsort()[::-1]]  # Sort results by score
    if cutoff!= None:
        contacts = results[1] <= cutoff
    else:
        contacts = results[1]>0
    not_contacts = ~contacts
    tpr = np.concatenate([[0], contacts.cumsum() / contacts.sum()])
    fpr = np.concatenate([[0], not_contacts.cumsum() / not_contacts.sum()])
    return np.array([fpr, tpr])


def compute_auc(roc):
    fpr, tpr = roc
    auc = np.sum(tpr[:-1] * (fpr[1:] - fpr[:-1]))
    return auc


def plot_roc(roc):
    import matplotlib.pyplot as plt
    plt.plot(roc[0], roc[1])
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (sensiticity)')
    plt.suptitle('Receiver operating characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '--')


def plot_singleresidue_decoy_energy(decoy_energy, native_energy, method='clustermap'):
    import seaborn as sns
    if method=='clustermap':
        f=sns.clustermap
    elif method == 'heatmap':
        f = sns.heatmap
    g = f(decoy_energy, cmap='RdBu_r',
          vmin=native_energy - decoy_energy.std() * 3,
          vmax=native_energy + decoy_energy.std() * 3)
    AA_dict = {str(i): _AA[i] for i in range(len(_AA))}
    new_ticklabels = []
    if method == 'clustermap':
        ax_heatmap = g.ax_heatmap
    else:
        ax_heatmap = g.axes
    for t in ax_heatmap.get_xticklabels():
        t.set_text(AA_dict[t.get_text()])
        new_ticklabels += [t]
    ax_heatmap.set_xticklabels(new_ticklabels)
    return g


def write_tcl_script(pdb_file, chain, mask, distance_matrix, distance_cutoff, single_frustration, pair_frustration, tcl_script='frustration.tcl',
                     max_connections=None, movie_name=None):
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration,nan=0,posinf=0,neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration,nan=0,posinf=0,neginf=0)
    
    
    structure = prody.parsePDB(str(pdb_file))
    selection = structure.select('protein', chain=chain)
    residues = np.unique(selection.getResnums())

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    for r, f in zip(residues, single_frustration):
        # print(f)
        fo.write(f'[atomselect top "chain {chain} and residue {int(r)}"] set beta {f}\n')

    # Mutational frustration:
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),distance_matrix.ravel(), mask.ravel()]).T
    #Filter with mask and distance
    if distance_cutoff:
        mask_dist=(sel_frustration[:, -2] <= distance_cutoff)
    else:
        mask_dist=np.ones(len(sel_frustration),dtype=bool)
    sel_frustration = sel_frustration[mask_dist & (sel_frustration[:, -1] > 0)]
    
    minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -0.78]
    sort_index = np.argsort(minimally_frustrated[:, 2])
    minimally_frustrated = minimally_frustrated[sort_index]
    if max_connections:
        minimally_frustrated = minimally_frustrated[:max_connections]
    fo.write('draw color green\n')
    

    for (r1, r2, f, d ,m) in minimally_frustrated:
        r1=int(r1)
        r2=int(r2)
        pos1 = selection.select(f'resid {r1} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        pos2 = selection.select(f'resid {r2} and chain {chain} and (name CB or (resname GLY and name CA))').getCoords()[0]
        distance = np.linalg.norm(pos1 - pos2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= distance <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')

    frustrated = sel_frustration[sel_frustration[:, 2] > 1]
    sort_index = np.argsort(frustrated[:, 2])[::-1]
    frustrated = frustrated[sort_index]
    if max_connections:
        frustrated = frustrated[:max_connections]
    fo.write('draw color red\n')
    for (r1, r2, f ,d, m) in frustrated:
        r1=int(r1)
        r2=int(r2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        if 3.5 <= d <= 6.5:
            fo.write(f'draw line $pos1 $pos2 style solid width 2\n')
        else:
            fo.write(f'draw line $pos1 $pos2 style dashed width 2\n')
    
    fo.write('''mol delrep top 0
            mol color Beta
            mol representation NewCartoon 0.300000 10.000000 4.100000 0
            mol selection all
            mol material Opaque
            mol addrep top
            color scale method GWR
            ''')
    
    if movie_name:
        fo.write('''axes location Off
            color Display Background white
            display resize 800 800
            display projection Orthographic
            display depthcue off
            display resetview
            display resize [expr [lindex [display get size] 0]/2*2] [expr [lindex [display get size] 1]/2*2] ;#Resize display to even height and width
            display update ui

            # Set up the movie directory and base file name
            mkdir movie_tmp
            set workdir "movie_tmp"
            ''' + f'set basename "{movie_name}"' + '''
            set numframes 360
            set framerate 25

            # Function to rotate the molecule and capture frames
            proc captureFrames {} {
                global workdir basename numframes
                for {set i 0} {$i < $numframes} {incr i} {
                    # Rotate the molecule around the Y-axis
                    rotate y by 1
                    
                    # Capture the frame
                    set output [format "%s/$basename.%05d.tga" $workdir $i]
                    render snapshot $output
                }
            }

            # Function to convert frames to MP4
            proc convertToMP4 {} {
                global workdir basename numframes framerate

                set mybasefilename [format "%s/%s" $workdir $basename]
                set outputFile [format "%s.mp4" $basename]
                
                # Construct and execute the ffmpeg command
                
                set command "ffmpeg -y -framerate $framerate -i $mybasefilename.%05d.tga -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $outputFile"
                puts "Executing: $command"
                exec ffmpeg -y -framerate $framerate -i $mybasefilename.%05d.tga -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $outputFile >&@ stdout
            }

            # Main script execution
            captureFrames
            convertToMP4

            # Cleanup the TGA files if desired
            for {set i 0} {$i < $numframes} {incr i} {
                set output [format "%s/$basename.%05d.tga" $workdir $i]
                exec rm $output
            }
            exit
        ''')
    fo.close()
    return tcl_script


def call_vmd(pdb_file, tcl_script):
    import subprocess
    return subprocess.Popen(['vmd', '-e', tcl_script, pdb_file], stdin=subprocess.PIPE)


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


