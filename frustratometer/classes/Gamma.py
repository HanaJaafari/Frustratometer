import numpy as np
import json
import copy
from pathlib import Path

class Gamma:
    default_alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    default_segment_definition = {
        'Burial': (3, 20),
        'Direct': (1, 20, 20),
        'Protein': (1, 20, 20),
        'Water': (1, 20, 20),
        
    }

    # Initialization
    def __init__(self, data, segment_definition=None, description=None, alphabet=None):
        # Depending on the type of 'data', different initialization methods are called.
        if isinstance(data, np.ndarray):
            self._init_from_array(data)
        elif isinstance(data, Gamma):
            self._init_from_instance(data)
        elif isinstance(data, Path) and data.exists():
            self._init_from_file(data)
        elif isinstance(data, str):
            self._init_from_file(data)
        else:
            raise TypeError("Unsupported type for initializing Gamma.")
        print(self.gamma_array)
        
        self.alphabet = alphabet if alphabet is not None else self.default_alphabet.copy()
        self.segment_definition = segment_definition if segment_definition is not None else self.default_segment_definition.copy()
        self.description = description

        self._validate_segments()

    def _init_from_array(self, gamma_array):
        self.gamma_array = gamma_array

    def _init_from_instance(self, gamma_object):
        self.gamma_array=gamma_object.gamma_array.copy()
        self.segment_definition = gamma_object.segment_definition.copy()
        self.alphabet = gamma_object.alphabet[:]
        self.description = gamma_object.description
        
    def _init_from_file(self, filepath):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        with open(filepath, 'r') as file:
            data = json.load(file)
        self.gamma_array = np.array(data["gamma_array"])
        self.description = data.get("description")
        file_segment_definition = data.get("segment_definition", {})
        self.segment_definition = {key: tuple(value) for key, value in file_segment_definition.items()}
        self.alphabet = data.get("alphabet")

    def _validate_segments(self):
        # Validate segment definitions against the gamma array shape.
        expected_size = sum(count * np.prod(shape) for _, (count, *shape) in self.segment_definition.items())
        if expected_size != self.gamma_array.size:
            raise ValueError(f"Expected gamma array size to be {expected_size}, got {self.gamma_array.size}.")

    @classmethod
    def from_array(cls, gamma_array, segment_definition=None, description=None, alphabet=None):
        """Create a Gamma instance from a numpy array and optional segment definition, description, and alphabet."""
        return cls(gamma_array, segment_definition, description, alphabet)

    @classmethod
    def from_instance(cls, gamma_object):
        """Create a new Gamma instance as a copy of an existing Gamma instance."""
        # This method creates a deep copy of the provided Gamma instance,
        # including copying its numpy array to ensure the new instance is independent.
        new_instance = copy.deepcopy(gamma_object)
        return new_instance

    #JSON support
    @classmethod
    def from_file(cls, filepath, segment_definition=None, alphabet=None):
        """Create a Gamma instance from a file path, optionally overriding the segment definition and alphabet."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        with path.open('r') as file:
            data = json.load(file)

        # Ensure gamma_array is of type float64, raise error if conversion is not possible
        try:
            gamma_array = np.array(data["gamma_array"], dtype=np.float64)
        except ValueError as e:
            raise TypeError(f"Gamma array contains non-numeric data that cannot be converted to float64: {e}")

        description = data.get("description")
        
        # Use the file's segment definition and alphabet if not overridden and if available
        file_segment_definition = data.get("segment_definition", {})
        if segment_definition is None:
            segment_definition = {key: tuple(value) for key, value in file_segment_definition.items()}
        alphabet = alphabet if alphabet is not None else data.get("alphabet")
        
        return cls(gamma_array, segment_definition, description, alphabet)

    def to_file(self, filepath: str, allow_overwrite: bool = False):
        """
        Saves the instance in JSON format.

        Args:
            filepath (str): The path to the file where the Gamma object will be saved.
            allow_overwrite (bool): Whether to allow overwriting if the file already exists. Default is False.

        Returns:
            None
        """
        
        if not allow_overwrite and Path(filepath).exists():
            raise FileExistsError(f"File '{filepath}' already exists. Set 'allow_overwrite' to True to overwrite.")

        data = {
            "gamma_array": self.gamma_array.tolist(),
            "segment_definition": self.segment_definition,
            "description": self.description,
            "alphabet": self.alphabet
        }
        with open(filepath, 'w') as file:
            json.dump(data, file)

        return filepath

    #AWSEM files support
    @classmethod
    def from_awsem_files(cls, 
                         burial_gamma_filepath='/home/cb/Development/awsemml/awsemml/data/awsem_parameters/burial_gamma.dat', 
                         gamma_dat_filepath='/home/cb/Development/awsemml/awsemml/data/awsem_parameters/gamma.dat'):

        # Transpose burial gamma data to align with the convention (3 rows by 20 columns)
        burial_gamma = np.loadtxt(burial_gamma_filepath).T

        # Load and reshape gamma.dat data
        contact_gamma_data = np.loadtxt(gamma_dat_filepath)
        gamma_compact=contact_gamma_data.reshape(2,210,2).transpose(1,0,2).reshape(210,4).T
        
        # Initialize container for expanded matrices
        expanded_matrices = [burial_gamma]

        # Expand each matrix from its compact form
        for i in [0,2,3]:
            compact_matrix = gamma_compact[i]
            full_matrix = np.zeros((20, 20))
            ix,jx = np.triu_indices_from(full_matrix)
            full_matrix[ix,jx] = compact_matrix
            full_matrix[jx,ix]=compact_matrix
            expanded_matrices.append(full_matrix)

        # Correctly combine burial gamma with expanded matrices
        gamma_array = np.concatenate([mat.ravel() for mat in expanded_matrices])
        return Gamma(gamma_array, description='AWSEM-MD(2015) gamma')

    def to_awsem_files(self,  burial_gamma_path, contact_gamma_path):
        """Save the gamma array to AWSEM files."""
        burial_gamma_data = self.gamma_array[:60]
        contact_gamma_data = self.gamma_array[60:]
        
        burial_gamma = burial_gamma_data.reshape((3, 20)).T
        contact_gamma = self.compact(contact_gamma_data.reshape(3,20,20))
        contact_gamma = contact_gamma[[0,0,1,2]]
        contact_gamma = contact_gamma.T.reshape(210,2,2).transpose(1,0,2).reshape(420,2)
        print(contact_gamma)

        np.savetxt(burial_gamma_path, burial_gamma, fmt='% .5f', delimiter='  ')
        np.savetxt(contact_gamma_path, contact_gamma, fmt='% .5f', delimiter='  ')

    
    #Deepcopy
    def deepcopy(self):
        """Creates a deep copy of this Gamma instance."""
        # Utilize the __init__ method to ensure a deep copy of each attribute
        return Gamma(self.gamma_array.copy(), 
                     segment_definition=copy.deepcopy(self.segment_definition), 
                     alphabet = self.alphabet[:],
                     description=self.description
                    )


    #Dividing and reordering
    def reorder(self, alphabet=None, segment_definition=None):
        # Create a deep copy of the current instance to avoid in-place modifications
        new_instance = self.deepcopy()
        
        # Perform reordering on the new instance
        if alphabet:
            if len(alphabet) != len(new_instance.alphabet):
                raise ValueError("The new alphabet must have the same length as the current alphabet.")
            if set(alphabet) != set(new_instance.alphabet):
                raise ValueError("The new alphabet includes different letters than the current alphabet.")
            if alphabet != new_instance.alphabet:
                new_instance._reorder_alphabet(alphabet)
        
        
        if segment_definition:
            if set(segment_definition) != set(new_instance.segment_definition.keys()):
                raise ValueError("The new segment order must contain the same segments as the current segment definition.")
            if segment_definition != list(new_instance.segment_definition.keys()):
                new_instance._reorder_segments(segment_definition)
        
        return new_instance

    def _reorder_alphabet(self, new_order):
        amino_acid_order = {aa: i for i, aa in enumerate(self.alphabet)}
        ordered_indices = [amino_acid_order[aa] for aa in new_order]
        
        segments = self.divide_into_segments()
        reordered_segments = []
        
        for name, segment in segments.items():
            shape=segment.shape[1:]
            if len(shape) == 1:
                reordered_segment = segment[:, ordered_indices]
                reordered_segments.append(reordered_segment.flatten())
            elif len(shape) == 2:
                reordered_segment = segment[:,ordered_indices,:][:,:, ordered_indices]
                reordered_segments.append(reordered_segment.flatten())
        
        # Update the instance's gamma_array and alphabet without modifying the original
        self.gamma_array = np.concatenate(reordered_segments)
        self.alphabet = new_order

    def _reorder_segments(self, new_segment_order):
        segments = self.divide_into_segments()
        new_gamma_array = []

        for segment_name in new_segment_order:
            segment_data = segments[segment_name]
            new_gamma_array.append(segment_data.flatten())

        self.gamma_array = np.concatenate(new_gamma_array)
        self.segment_definition = new_segment_order

    def divide_into_segments(self, split=False):
        '''Split the gamma_array based on the current segment definitions'''
        segments = {}
        
        start = 0
        for name, (count, *shape) in self.segment_definition.items():
            
            if split:
                for i in range(count):
                    segment_name=name
                    if count>1:
                        segment_name=f'{name}_{i}'
                    end = start + np.prod(shape)
                    segment = self.gamma_array[start:end].reshape(tuple(shape))
                    segments[segment_name]=segment
                    start = end
            else:
                end = start + count * np.prod(shape) 
                segment = self.gamma_array[start:end].reshape((count,*shape))
                segments[name]=segment
                start = end
        return segments
    
    #Compact and expand segments to account for symmetrical 2D matrices
    @staticmethod
    def compact(matrix,sum_offdiagonal=False):
        """Compacts a 2D symmetric matrix into a 1D array."""
        if matrix.ndim != 3 or matrix.shape[-2] != matrix.shape[-1]:
            return matrix
        if sum_offdiagonal:
            matrix=matrix.copy()
            compact=[]
            for m in matrix:
                ix=np.triu_indices(m.shape[-1],k=1)
                iy=ix[1],ix[0]
                m[ix]+=m[iy]
                compact+=[m[np.triu_indices(m.shape[-1])]]
            return np.array(compact)
        else:            
            return np.array([m[np.triu_indices(m.shape[-1])] for m in matrix])

    @staticmethod
    def expand(compacted_array,sum_offdiagonal=False):
        # Calculate the potential size of the symmetric matrix
        compact_size = len(compacted_array)
        expanded_size = int((-1 + np.sqrt(1 + 8 * compact_size)) / 2) # From n*(n+1)/2
        if (expanded_size * (expanded_size + 1)) / 2 != compact_size:
            raise ValueError("The size of the compacted array does not match a valid symmetric matrix.")
        
        # Initialize the symmetric matrix
        matrix = np.zeros((expanded_size, expanded_size))
        rows, cols = np.triu_indices(expanded_size)
        if sum_offdiagonal==True:
            matrix[rows, cols] += compacted_array/2  # Fill lower part
            matrix[cols, rows] += compacted_array/2 # Fill upper part
        else:
            matrix[rows, cols] = compacted_array  # Fill lower part
            matrix[cols, rows] = compacted_array  # Fill upper part
        return matrix
    
    def compact_segments(self, sum_offdiagonal=False):
        """Compacts all 2D segments of the gamma array, keeping only the upper triangle including the diagonal."""
        compact_size={n:int(c*(s[0]+1)*s[0]/2 if len(s)==2 else c*s[0]) for n,(c,*s) in self.segment_definition.items()}
        compacted = np.zeros(np.sum(list(compact_size.values())))
        start=0
        for name, segment in self.divide_into_segments().items():
            end=start + compact_size[name]
            compacted[start:end]=self.compact(segment,sum_offdiagonal=sum_offdiagonal).ravel() if segment.ndim==3 else segment.ravel()
            start=end
        return compacted
    
    def expand_segments(self, compacted, sum_offdiagonal=False):
        """
        Expands the compacted Gamma array segments into their full forms and returns a new Gamma instance.
        """
        
        compact_size={n:int(c*(s[0]+1)*s[0]/2 if len(s)==2 else c*s[0]) for n,(c,*s) in self.segment_definition.items()}
        assert len(compacted) == np.sum(list(compact_size.values()))
        
        # Initialize an empty list to hold the expanded segments
        size=np.sum([np.prod(s) for s in self.segment_definition.values()])
        expanded = np.zeros(size)

        # Expand each segment
        start_compacted=0
        start_expanded=0
        for name,(count, *shape) in self.segment_definition.items():
            for i in range(count):
                if len(shape)==1:
                    expanded_size=compact_size=shape[0]
                elif len(shape)==2:
                    expanded_size=shape[0]**2
                    compact_size=((shape[0]+1)*shape[0])//2
                else:
                    raise ValueError
                end_compacted = start_compacted + compact_size
                end_expanded  = start_expanded  + expanded_size
                expanded[start_expanded:end_expanded]=self.expand(compacted[start_compacted:end_compacted],sum_offdiagonal=sum_offdiagonal).ravel() if len(shape)==2 else compacted[start_compacted:end_compacted]
                start_compacted=end_compacted
                start_expanded =end_expanded

        new_instance = self.deepcopy()
        new_instance.gamma_array = expanded
        # Return a new Gamma instance with the expanded array
        return new_instance

    def normalize(self):
        new_gamma=self.deepcopy()
        new_gamma.gamma_array/=np.linalg.norm(new_gamma.gamma_array)
        return new_gamma

    def center(self):
        new_gamma=self.deepcopy()
        new_gamma.gamma_array-=new_gamma.gamma_array.mean()
        return new_gamma

    def symmetrize(self):
        new_gamma=self.deepcopy()
        segments = self.divide_into_segments()
        start=0
        for name,segment in segments.items():
            end=start+np.prod(segment.size)
            if len(segment.shape) == 3:
                symmetrized_segment = (segment + segment.transpose(0, 2, 1)) / 2
                new_gamma.gamma_array[start:end]= symmetrized_segment.ravel()
            else:
                new_gamma.gamma_array[start:end]= segment.ravel()
            start=end
        return new_gamma

    # Correlations
    def correlate(self, other):
        """Compares the compacted form of self with other, ensuring they are compatible."""
        other = Gamma(other)
        other.reorder(alphabet=self.alphabet,segment_definition=self.segment_definition)        
        # Compact both self and other before correlation
        self_compacted = self.compact_segments()
        other_compacted = other.compact_segments()
        
        # Compute correlation on compacted arrays
        correlation_result = np.corrcoef(self_compacted, other_compacted)[0, 1]
        return correlation_result

    def correlate_segments(self, other):
        """Correlates individual segments between self and other, using compacted forms for 2D segments."""
        other = Gamma(other)
        other.reorder(alphabet=self.alphabet,segment_definition=self.segment_definition)

        self_segments = self.divide_into_segments()
        other_segments = other.divide_into_segments()

        correlations = {}
        for name in self_segments:
            if name not in other_segments:
                continue
            seg1 = self.compact(self_segments[name])
            seg2 = self.compact(other_segments[name])
            # Flatten if they are lists (multiple segments)
            correlation = np.corrcoef(seg1.flatten(), seg2.flatten())[0, 1]
            correlations[name]=correlation
        return correlations
    


    # Plotting
    def plot_gamma(self, new_order=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        if new_order:
            self._reorder_alphabet(new_order)
        segments = self.divide_into_segments()
        
        # Plot setup
        f, axes = plt.subplots(2, 2, figsize=(18, 16))
        titles = ['Burial Gammas', 'Direct Gammas', 'Water Gammas', 'Protein Gammas']

        for i, (title, name) in enumerate(zip(titles, segments)):
            ax = axes[i // 2, i % 2]
            sns.heatmap(segments[name].reshape(-1, 20), ax=ax, cmap='RdBu_r', center=0)
            ax.set_title(title)
            ax.set_xticks(np.arange(len(self.alphabet)) + 0.5)
            ax.set_xticklabels(self.alphabet)
            ax.set_yticks(np.arange(segments[name].shape[0] // 20) + 0.5)
            ax.set_yticklabels(range(segments[name].shape[0] // 20))

        plt.tight_layout()
        plt.show()

    def compare_parameters(self,other, linthresh = 0.01, grid=True):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        split = True
        g1=self.deepcopy()
        g2=other.deepcopy()
        g1-=g1.gamma_array.mean()
        g2-=g2.gamma_array.mean()
        g1/=np.linalg.norm(g1.gamma_array)
        g2/=np.linalg.norm(g2.gamma_array)

        alphabet = self.alphabet
        segments1=g1.divide_into_segments(split=True)
        segments2=g2.divide_into_segments(split=True)

        # Enhanced polarity categories with a simple division for non-polar
        polarity_colors = {
            # Nonpolar, more hydrophobic
            'V': 'black', 'L': 'black', 'I': 'black', 'P': 'black', 'W': 'black', 'F': 'black', 'M': 'black',
            # Nonpolar, less hydrophobic
            'A': 'darkgray', 'G': 'darkgray', 'C': 'darkgray',
            # Polar uncharged
            'S': 'green', 'T': 'green', 'Y': 'green', 'N': 'green', 'Q': 'green',
            # Charged positive
            'K': 'red', 'R': 'red', 'H': 'red',
            # Charged negative
            'D': 'blue', 'E': 'blue'
        }

        # Function to determine the interaction type
        def get_interaction_type(aa1, aa2):
            polar = {'S', 'T', 'Y', 'N', 'Q', 'K', 'R', 'H', 'D', 'E'}
            nonpolar = {'A', 'V', 'L', 'I', 'P', 'W', 'F', 'M', 'G', 'C'}
            
            if aa1 in polar and aa2 in polar:
                return 'green'  # Polar-Polar
            elif aa1 in nonpolar and aa2 in nonpolar:
                return 'black'  # Nonpolar-Nonpolar
            else:
                return 'red'  # Polar-Nonpolar

        for name in segments1.keys():
            fig, ax = plt.subplots(figsize=(8,8))
            ax.scatter(segments1[name], segments2[name], s=0.1)
            if len(segments1[name].shape) == 1:
                for i, aa1 in enumerate(alphabet):
                    ax.text(segments1[name][i], segments2[name][i], aa1, color=polarity_colors[aa1], size=15, ha='center', va='center')
            elif len(segments1[name].shape) == 2:
                for i, aa1 in enumerate(alphabet):
                    for j, aa2 in enumerate(alphabet):
                        if j >= i:
                            ax.text(segments1[name][i,j], segments2[name][i,j], f'{aa1}-{aa2}', color=get_interaction_type(aa1, aa2), size=10, ha='center', va='center')


            max_val = max(-g1.gamma_array.min(), g1.gamma_array.max(), -g2.gamma_array.min(), g2.gamma_array.max())
            ax.set_xlim(-max_val*1.2, max_val*1.2)
            ax.set_ylim(-max_val*1.2, max_val*1.2)
            
            
            if linthresh:
                # This value might need adjustment based on your data
                ax.set_xscale('symlog', linthresh=linthresh)
                ax.set_yscale('symlog', linthresh=linthresh)
                if grid:
                    ax.grid(True, which="both", ls="--")  # Optional: add grid for better visibility of scales

                # Set major and minor ticks for symlog scale
                ax.xaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10,linthresh=linthresh))
                ax.yaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10,linthresh=linthresh))
                ax.xaxis.set_minor_locator(ticker.SymmetricalLogLocator(subs=[.25,.5,.75],base=10,linthresh=linthresh))
                ax.yaxis.set_minor_locator(ticker.SymmetricalLogLocator(subs=[.25,.5,.75],base=10,linthresh=linthresh))

            ax.set_title(name)
            ax.set_xlabel(g1.description)
            ax.set_ylabel(g2.description)
            
            return ax
    

    # Dunder methods
    def as_array(self):
        """Return the internal gamma_array as a NumPy array."""
        return self.gamma_array
    
    def __len__(self):
        return len(self.gamma_array)

    def __setitem__(self, key, value):
        self.gamma_array[key] = value

    def __repr__(self):
        return f"Gamma(array={self.gamma_array}, description='{self.description}')"

    def __add__(self, other):
        if isinstance(other, Gamma):
            return Gamma(self.gamma_array + other.gamma_array, self.segment_definition, self.description)
        else:
            return Gamma(self.gamma_array + other, self.segment_definition, self.description)

    def __sub__(self, other):
        if isinstance(other, Gamma):
            return Gamma(self.gamma_array - other.gamma_array, self.segment_definition, self.description)
        else:
            return Gamma(self.gamma_array - other, self.segment_definition, self.description)

    def __mul__(self, other):
        if isinstance(other, Gamma):
            return Gamma(self.gamma_array * other.gamma_array, self.segment_definition, self.description)
        else:
            return Gamma(self.gamma_array * other, self.segment_definition, self.description)

    def __truediv__(self, other):
        if isinstance(other, Gamma):
            return Gamma(self.gamma_array / other.gamma_array, self.segment_definition, self.description)
        else:
            return Gamma(self.gamma_array / other, self.segment_definition, self.description)

    # Add the reciprocal magic method for division operations
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, Gamma):
            return Gamma(other.gamma_array - self.gamma_array, self.segment_definition, self.description)
        else:
            return Gamma(other.gamma_array - self, self.segment_definition, self.description)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Gamma(other / self.gamma_array, self.segment_definition, self.description)

    # It might also be wise to implement the in-place arithmetic operations
    def __iadd__(self, other):
        self.gamma_array = self.__add__(other).gamma_array
        return self

    def __isub__(self, other):
        self.gamma_array = self.__sub__(other).gamma_array
        return self

    def __imul__(self, other):
        self.gamma_array = self.__mul__(other).gamma_array
        return self

    def __itruediv__(self, other):
        self.gamma_array = self.__truediv__(other).gamma_array
        return self
    
    def get_segment(self, segment_name):
        """
        Retrieve a segment by its name with the correct shape as defined in segment_definition.
        """
        if segment_name not in self.segment_definition:
            raise KeyError(f"Segment '{segment_name}' is not defined.")
        
        segments = self.divide_into_segments(split=False)
        if segment_name in segments:
            return segments[segment_name]

    def __getitem__(self, key):
        """
        Allow dictionary-like access to retrieve segments by their names.
        """
        if isinstance(key, int):
            return self.gamma_array[key]
        else:
            return self.get_segment(key)
    
    def __setitem__(self, key, value):
        """
        Allow dictionary-like assignment to modify segments by their names.
        """
        if key not in self.segment_definition:
            raise KeyError(f"Segment '{key}' is not defined.")

        # Calculate the starting index for the segment in the gamma_array
        start_index = 0
        for segment_name, (count, *shape) in self.segment_definition.items():
            if segment_name == key:
                expected_size = np.prod(shape) * count
                if value.size != expected_size:
                    raise ValueError(f"Size mismatch. Expected {expected_size}, got {value.size}.")
                
                end_index = start_index + expected_size
                # Reshape value to match the segment shape before assignment
                self.gamma_array[start_index:end_index] = value.reshape(-1)
                break
            else:
                start_index += np.prod(shape) * count

if __name__ == '__main__':
    #unittest.main()
    
    class O():
        pass
    self=O()
    self.segment_definition = {
        '1D': (2, 5),  # Two 1D segments of length 5
        '2D': (2, 5, 5)  # Two 2D segments of shape 5x5
    }
    self.gamma_array = np.arange(2*5 + 2*5*5, dtype=np.float64)  # Array to match the segment definition
    self.gamma = Gamma(self.gamma_array, segment_definition=self.segment_definition, description="Test Gamma", alphabet=['A', 'B', 'C', 'D', 'E'])
    self.gamma['2D'] = (self.gamma['2D'] + self.gamma['2D'].transpose(0,2,1))

    print(self.gamma.gamma_array.sum())
    # Call the normalize function
    normalized_gamma = self.gamma.normalize()
    print(self.gamma.gamma_array.sum())

    # Check if the gamma_array is normalized
    np.testing.assert_almost_equal((normalized_gamma.gamma_array**2).sum(), 1)

    awsem_gamma=Gamma.from_awsem_files()
    awsem_gamma.to_awsem_files('burial_test.dat','contact_test.dat')

    test=Gamma(np.arange(1260))
    test.plot_gamma()

    test_sym=test.symmetrize()
    test_sym.plot_gamma()

    self.gamma1 = Gamma(np.arange(0,1260,1))
    self.gamma2 = Gamma(np.arange(0,1260,1)*5+10)
    self.gamma3 = Gamma(np.arange(1260,0,-1)*2-4)