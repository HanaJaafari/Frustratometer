import scipy.io
import subprocess

def compute_plm(protein_name, reweighting_threshold=0.1, nr_of_cores=1):
    """
    Calculate Potts Model Fields and Couplings Terms
    :param protein_name
    :param reweighting_threshold
    :param nr_of_cores
    """
    '''Returns matrix consisting of Potts Model Fields and Couplings terms'''
    # MATLAB needs Bioinfomatics toolbox and Image processing toolbox to parse the sequences
    # Functions need to be compiled with 'matlab -nodisplay -r "mexAll"'
    # See: https://www.mathworks.com/help/matlab/call-mex-functions.html
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h' % _path, nargout=0)
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h/functions' % _path, nargout=0)
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h/3rd_party_code/minFunc' % _path, nargout=0)
        print('plmDCA_asymmetric', protein_name, reweighting_threshold, nr_of_cores)
        eng.plmDCA_asymmetric(protein_name, reweighting_threshold, nr_of_cores, nargout=0)  # , stdout=out )
    except ImportError:
        subprocess.call(['matlab', '-nodisplay', '-r',
                         f"plmDCA_asymmetric({protein_name},{reweighting_threshold},{nr_of_cores},nargout=0);quit"])

def load_potts_model(potts_model_file):
    return scipy.io.loadmat(potts_model_file)