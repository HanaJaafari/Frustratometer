from Bio import Align
import numpy as np

class Map():
    def __init__(self, map_array):
        self.map_array = map_array
        self.seq1_len = max(map_array[0]) + 1
        self.seq2_len = max(map_array[1]) + 1

    @classmethod
    def from_path(cls, path):
        total_size=(np.diff(path,axis=0).max(axis=1)).sum()
        ixs=np.full([2,total_size],-1,dtype=int)

        prev_i, prev_j = path[0]
        start=0
        for curr_i, curr_j in path[1:]:
            print(curr_i,curr_j,start,total_size)
            ixs[0,start:start+curr_i-prev_i]=np.arange(prev_i,curr_i)
            ixs[1,start:start+curr_j-prev_j]=np.arange(prev_j,curr_j)
            start=start+max(curr_i-prev_i,curr_j-prev_j)
            prev_i, prev_j=curr_i, curr_j
        return cls(ixs)

    @classmethod
    def from_sequences(cls, sequence_a, sequence_b, match_score=2, mismatch_score=-1, 
                       open_gap_score=-0.5, extend_gap_score=-0.1, target_end_gap_score=-0.01, query_end_gap_score=-0.01):
        
        aligner=Align.PairwiseAligner()
        aligner.match_score = match_score
        aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
        aligner.mismatch_score = mismatch_score
        aligner.open_gap_score = open_gap_score
        aligner.extend_gap_score = extend_gap_score
        aligner.target_end_gap_score = target_end_gap_score
        aligner.query_end_gap_score = query_end_gap_score
        
        alignments = aligner.align(sequence_a, sequence_b)
        alignment = sorted(alignments)[0]
        print("Score = %.1f:" % alignment.score) # Takes only first possible alignment
        print(alignment)
        return cls.from_path(alignment.path)
            
    def map(self, sequence=None, reverse=False):
        seq=np.array([a for a in sequence+'-'])
        if reverse:
            if len(sequence)!=self.seq1_len:
                raise IndexError(f'Sequence length ({len(sequence)}) does not match map ({self.seq1_len})')
            return ''.join(seq[self.map_array[0]][self.map_array[1]>-1]) #Second sequence to first sequence
        else:
            if len(sequence)!=self.seq2_len:
                print(sequence)
                print(self.map_array)
                raise IndexError(f'Sequence length ({len(sequence)}) does not match map ({self.seq2_len})')
                
            return ''.join(seq[self.map_array[1]][self.map_array[0]>-1]) #First sequence to second sequence
        
    def reverse(self):
        return self.__class__(self.map_array[::-1])
    
    def copy(self):
        return self.__class__(self.map_array.copy())
        
    def __repr__(self):
        """
        Provides a string representation of the SequenceMapper object, showing the sequences,
        alignment, and mappings.
        """
        s1='Seq1: '+''.join(['S' if i>-1 else '-' for i in self.map_array[0]])
        mm='Map:  '+''.join(['|' if ((i>-1) and (j>-1)) else '-' for i,j in self.map_array.T])
        s2='Seq2: '+''.join(['S' if i>-1 else '-' for i in self.map_array[1]])
        
        return f"{self.__class__}\n{s1}\n{mm}\n{s2}"

    @property
    def map_array(self):
        return self._map_array
    
    @map_array.setter
    def map_array(self, value):
        self._map_array = value
        self.seq1_len = max(value[0]) + 1
        self.seq2_len = max(value[1]) + 1


    
    

