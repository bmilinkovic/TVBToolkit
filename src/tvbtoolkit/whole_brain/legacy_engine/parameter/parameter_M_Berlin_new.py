import os

class Parameters:
    def __init__(self, 
         parameter_simulation=None,
         parameter_model=None, 
         parameter_connection_between_region=None, 
         parameter_coupling=None, 
         parameter_integrator=None,
         parameter_monitor=None,
         parameter_stimulus=None,
         **kwargs):
        
        if parameter_simulation is not None:
            self.parameter_simulation = parameter_simulation
            self.parameter_model = parameter_model
            self.parameter_connection_between_region = parameter_connection_between_region
            self.parameter_coupling = parameter_coupling
            self.parameter_integrator = parameter_integrator
            self.parameter_monitor = parameter_monitor
            self.parameter_stimulus = parameter_stimulus
            return
            
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./result/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            'matteo':False,
            'gK_gNa':False,
            #order of the model
            'order':2,
            #parameter of the model
            'inh_factor':1.,
            'E_Na_e':50.,
            'E_Na_i':50.,
            'E_K_e':-90.,
            'E_K_i':-90.,
            'g_L':10.0,
            'g_K_e':8.214285714285714,
            'g_Na_e':1.7857142857142865,
            'g_K_i':8.214285714285714,
            'g_Na_i':1.7857142857142865,
            'E_L_e':-64.0,
            'E_L_i':-65.0,
            'C_m':200.0,
            'b_e':5.0, # 60
            'a_e':0.0, # 4.0
            'b_i':0.0,
            'a_i':0.0,
            'tau_w_e':500.0,
            'tau_w_i':1.0,
            'E_e':0.0,
            'E_i':-80.0,
            'Q_e':1.5, # 1.25
            'Q_i':5.0,
            'tau_e_e':5.0,
            'tau_e_i':5.0,
            'tau_i':5.0,
            'N_tot':10000,
            'p_connect_e':0.05,
            'p_connect_i':0.05,
            'g':0.2,
            'T':20.0, # 5
            # I am changing this to put Fede's config, to see if it works better
            'P_e':[-0.05017034,  0.00451531, -0.00794377, -0.00208418, -0.00054697,
        0.00341614, -0.01156433,  0.00194753,  0.00274079, -0.01066769],
            'P_i':[-0.05184978,  0.0061593 , -0.01403522,  0.00166511, -0.0020559 ,
        0.00318432, -0.03112775,  0.00656668,  0.00171829, -0.04516385],
#            Presumably previous config:
#            'T':40.0,
#            'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
#            'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
            'external_input_ex_ex':0.315*1e-3, # KHz
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            'tau_OU':5.0,
            'weight_noise': 1e-4,#10.5*1e-5, #1e-4, #10.5*1e-5,
            'noise_alpha':0.0,  # 0: private-only, 1: shared-only
            'shared_noise_mode':'none',  # one of: none, global, connectivity
            'K_ext_e':400,
            'K_ext_i':0,
            #Initial condition :
            'initial_condition':{
                "E": [0.000, 0.000],"I": [0.00, 0.00],"C_ee": [0.0,0.0],"C_ei": [0.0,0.0],"C_ii": [0.0,0.0],"W_e": [100.0, 100.0],"W_i": [0.0,0.0],"noise":[0.0,0.0]}
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':True,
            'from_h5':False,
            'path':path+'/../../data/connectivity',#path+'/../../data/QL_20120814/', #the files
            'conn_name':'connectivity_68.zip',#'Connectivity.zip',
            # File description
            'number_of_regions':0, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':4.0,
            'normalised':True,
            'nullify_diagonals':True,
            # disconnect certain regions
            'disconnect_regions':[],
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'coupling_parameter':{
                    'a':0.3,
                    'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'parameter_Raw':{
                'variables_of_interest':[0,1]},
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4,5,6,7],
                'period':self.parameter_integrator['dt']*10.0},
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0], #only the excitatory
                'period':self.parameter_integrator['dt']*20000.0},
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1}
        }


        self.parameter_stimulus = {
            'stimtime': 99.0,
            "stimdur": 9.0,
            "stimperiod": 1e9,
            "stimregion": None,
            "stimval":0.,
            "stimvariables":[0]
        }
        
    def __eq__(self, other):
        if isinstance(other, Parameters):
            return (
#                    self.parameter_simulation == other.parameter_simulation and
                    self.parameter_model == other.parameter_model and
                    self.parameter_connection_between_region == other.parameter_connection_between_region and
                    self.parameter_coupling == other.parameter_coupling and
                    self.parameter_integrator == other.parameter_integrator and
                    self.parameter_monitor == other.parameter_monitor and
                    self.parameter_stimulus == other.parameter_stimulus
                    )
        if isinstance(other, dict):
            return (
                dict_inclusion_except(self.parameter_simulation, other['parameter_simulation'], ['path_result']) and
                self.parameter_model == other['parameter_model'] and
                self.parameter_connection_between_region == other['parameter_connection_between_region'] and
                self.parameter_coupling == other['parameter_coupling'] and
                self.parameter_integrator == other['parameter_integrator'] and
                self.parameter_monitor == other['parameter_monitor'] and
                self.parameter_stimulus == other['parameter_stimulus']
                )
        raise NotImplementedError
            
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
def dict_inclusion_except(d1, d2, exceptions):
    """
        This checks that all keys of dictionary d1 are in dictionary d2 except exceptions
    """
    for k,v in d1.items():
        if k in exceptions:
            pass
        if k not in d2:
            return False
    return True
