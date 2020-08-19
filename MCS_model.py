from utils import check_folder
import numpy as np
import scipy.io
import scipy.stats as st
import statsmodels as sm
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.patches as patches
import pickle

#==============================================================================#
# SCALING BY A RANGE
def scaling(x,l,u,operation):
    # scaling() scales or unscales the vector x according to the bounds
    # specified by u and l. The flag type indicates whether to scale (1) or
    # unscale (2) x. Vectors must all have the same dimension.
    
    if operation == 1:
        # scale
        x_out=(x-l)/(u-l)
    elif operation == 2:
        # unscale
        x_out = l + x*(u-l)
    
    return x_out

#==============================================================================#
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # DISTRIBUTIONS = [        
    #     st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #     st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #     st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #     st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #     st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #     st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
    #     st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #     st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #     st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #     st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    # ]

    # DISTRIBUTIONS = [     
    #     st.pearson3, st.johnsonsu, st.nct, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
    #     st.tukeylambda, st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
    #     st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
    #     st.alpha, st.norm
    # ]

    DISTRIBUTIONS = [     
        st.pearson3, st.johnsonsu, st.burr, st.mielke, st.genlogistic, st.fisk, st.t, 
        st.hypsecant, st.logistic, st.dweibull, st.dgamma, st.gennorm, 
        st.vonmises_line, st.exponnorm, st.loglaplace, st.invgamma, st.laplace, st.invgauss, 
        st.alpha, st.norm
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    sse_d = []; name_d = []
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # print(distribution.name)
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)

                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

                sse_d += [sse]
                name_d += [distribution.name]

        except Exception:
            pass
        
    sse_d, name_d = (list(t) for t in zip(*sorted(zip(sse_d, name_d))))
    
    return (best_distribution.name, best_params, name_d[:6])

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def plot_distribution(data, fun_name, label_name, n_bins, run, 
                      discrete = False, min_bin_width = 0, 
                      fig_swept = None, run_label = 'PDF', color = u'b',
                      dataXLim = None, dataYLim = None, constraint = None,
                      fit_distribution = True, handles = [], labels = []):

    if constraint is not None:
        data_cstr = [d - constraint for d in data]
        mean_data = np.mean(data_cstr)
        std_data = np.std(data_cstr)
    else:
        mean_data = np.mean(data)
        std_data = np.std(data)

    # Plot raw data
    fig0 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        d = max(min(np.diff(np.unique(np.asarray(data)))), min_bin_width)
        left_of_first_bin = min(data) - float(d)/2
        right_of_last_bin = max(data) + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)

    ax = fig0.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Plot for comparison
    fig1 = plt.figure(figsize=(6,5))

    if discrete:
        # discrete bin numbers
        plt.hist(data, bins, alpha=0.5, density=True)
    else:
        plt.hist(data, bins = n_bins, alpha=0.5, density=True)
    
    ax = fig1.gca()

    # Update plots
    ax.set_ylim(ax.get_ylim())
    ax.set_xlabel(label_name)
    ax.set_ylabel('Frequency')

    # Display
    if fig_swept is None:
        fig2 = plt.figure(figsize=(6,5))
    else:
        fig2 = fig_swept
    
    ax2 = fig2.gca()

    if discrete:
        data_bins = bins
    else:
        data_bins = n_bins

    # Fit and plot distribution
    if fit_distribution:

        best_fit_name, best_fit_params, best_10_fits = best_fit_distribution(data, data_bins, ax)

        best_dist = getattr(st, best_fit_name)
        print('Best fit: %s' %(best_fit_name.upper()) )
        # Make PDF with best params 
        pdf = make_pdf(best_dist, best_fit_params)
        pdf.plot(lw=2, color = color, label=run_label, legend=True, ax=ax2)

        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)

        handles = []; labels = []
    else:
        lgd = ax2.legend(handles, labels, fontsize = 9.0)

    if discrete:
        # discrete bin numbers
        ax2.hist(data, bins, color = color, alpha=0.5, density=True)
    else:
        ax2.hist(data, bins = n_bins, color = color, alpha=0.5, density=True)

    # plot constraint limits
    if constraint is not None:
        ax2.axvline(x=constraint, linestyle='--', linewidth='2', color='k')

    # Save plot limits
    if dataYLim is None and dataXLim is None:
        dataYLim = ax2.get_ylim()
        dataXLim = ax2.get_xlim()
    else:
        # Update plots
        ax2.set_xlim(dataXLim)
        ax2.set_ylim(dataYLim)

    ax2.tick_params(axis='both', which='major', labelsize=14) 
    ax2.set_xlabel(label_name, fontsize=14)
    ax2.set_ylabel('Probability density', fontsize=14)

    fig0.savefig('MCS_results/RAW_%s_r%i.pdf' %(fun_name,run), 
        format='pdf', dpi=100,bbox_inches='tight')
    
    if fig_swept is None:
        fig2.savefig('MCS_results/PDF_%s_r%i.pdf' %(fun_name,run), 
                format='pdf', dpi=100,bbox_inches='tight')

    if fig_swept is None:    
        plt.close('all')
    else:
        plt.close(fig0)
        plt.close(fig1)
    
    return dataXLim, dataYLim, mean_data, std_data

#==============================================================================#
# Main file
if __name__ == '__main__':

    run = 0

    n_samples = 1000
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = False

    # Model variables
    bounds = np.array([[45.0           , 155.0   ], # Axial Position
                       [2.0            , 20      ], # Stiff height
                       [20.0           , 155.0   ], # Stiff width
                       [-100.0         , 100.0   ], # T1
                       [-100.0         , 100.0   ], # T2
                       [-100.0         , 100.0   ], # T3
                       [-100.0         , 100.0   ]]) # T4

    fit_cond = False # Do not fit data
    run = 0 # starting point

    #===================================================================#
    # DOE levels
    n_var = 2; n_samples = 1000; n_steps = 5
    var_DOE = np.linspace(0.0,1.0,n_steps)
    var_DOE = scaling(var_DOE,bounds[n_var,0],bounds[n_var,1],2)

    same_axis = True
    if same_axis:
        fig_nsafety = plt.figure(figsize=(10,5))
    else:
        fig_nsafety = None

    auto_limits = True
    if auto_limits:
        dataXLim = dataYLim = None
    else:
        with open('MCS_results/MCS_data_limits.pkl','rb') as fid:
            dataXLim = pickle.load(fid)
            dataYLim = pickle.load(fid)

    mean_nsafety_runs = []; std_nsafety_runs = []

    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                            r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]

    #===================================================================#
    # Initialize
    handles_lgd = []; labels_lgd = [] # initialize legend

    if new_run:
        # New MCS
        run = 0
        # Resume MCS
        # run = 3
        # points = points[run:]
        # labels = labels[run:]

        # terminate MCS
        # run = 3
        # run_end = 3 + 1
        # points = points[run:run_end]
        # labels = labels[run:run_end]

    for var in var_DOE:

        legend_labels = ['Axial position ($x_1$) = %f mm' %(var),
                         'Thickness ($x_2$) = %f mm' %(var),
                         'Width ($x_3$) = %f mm' %(var)]

        legend_label = legend_labels[n_var] 
        
        filename = 'MCS_results/DOE_R%i.mat' %(run+1)
        mat = scipy.io.loadmat(filename) # get optitrack data
        data = mat['MCS_runs']
        data = data[:,0::]
        
        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor=colors[run], facecolor=colors[run], fill='None' ,alpha=0.5)
        
        handles_lgd += [a]
        labels_lgd += [legend_label]

        label_name = u'Stochastic objective function ($\hat{f}_{\mathbf{P}}(\mathbf{x})$)'
        fun_name = 'nsafety'

        dataXLim_out, dataYLim_out, mean_nsafety, std_nsafety = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_nsafety, run_label = legend_label, color = colors[run], 
            dataXLim = dataXLim, dataYLim = dataYLim, 
            fit_distribution = fit_cond, handles = handles_lgd, labels = labels_lgd)

        mean_nsafety_runs += [mean_nsafety]
        std_nsafety_runs += [std_nsafety]

        if not auto_limits:
            fig_nsafety.savefig('MCS_results/PDF_%s_r%i.pdf' %('nsafety', run + 1), 
                                    format='pdf', dpi=100,bbox_inches='tight')

        print('==============================================')
        print('Run %i stats:' %(run))
        print('mean n_safety: %f; std n_safety: %f' %(mean_nsafety,std_nsafety))
        print('==============================================')

        run += 1

    with open('MCS_results/MCS_data_limits.pkl','wb') as fid:
        pickle.dump(dataXLim_out,fid)
        pickle.dump(dataYLim_out,fid)

    with open('MCS_results/MCS_data_stats.pkl','wb') as fid:
        pickle.dump(mean_nsafety,fid)
        pickle.dump(std_nsafety,fid)

    if same_axis:
        fig_nsafety.savefig('MCS_results/PDF_%s.pdf' %('nsafety'), 
                                format='pdf', dpi=100,bbox_inches='tight')
        plt.show()