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
                      fig_swept = None, run_label = 'PDF', color = u'b', hatch_pattern = u'',
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
        # ax2.hist(data, bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data, bins, linewidth=2, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)
    else:
        # ax2.hist(data, bins = n_bins, color = color, alpha=0.5, label = 'data', density=True)
        ax2.hist(data, bins = n_bins, facecolor=color, 
                 hatch=hatch_pattern, edgecolor='k',fill=True, density=True)

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

    n_samples = 1000
    n_bins = 30 # for continuous distributions
    min_bin_width_i = 15 # for discrete distributions
    min_bin_width_f = 5 # for discrete distributions

    new_run = True

    # Model variables
    bounds = np.array([[45.0           , 155.0   ], # Axial Position
                       [2.0            , 20      ], # Stiff height
                       [20.0           , 155.0   ], # Stiff width
                       [-100.0         , 100.0   ], # T1
                       [-100.0         , 100.0   ], # T2
                       [-100.0         , 100.0   ], # T3
                       [-100.0         , 100.0   ]]) # T4


    fit_cond = False # Do not fit data
    color_mode = 'color' # Choose color mode (black_White)
    run = 0 # starting point

    #===================================================================#
    # R0 opts (old algorithm)
    
    # # Points to plot
    # opt_1 = np.array([0.00007426524509085878, 0.42380805903074963981, 0.02843359294084374031])
    # opt_1 = scaling(opt_1, -1*np.ones(3), 1*np.ones(3), 1) # Normalize variables between -1 and 1 back to 0 and 1
    # opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_1 = opt_1_unscaled[0] + opt_1_unscaled[2] - 201

    # opt_2 = np.array([0.06624374389648435280, 0.48429003953933713600, 0.14601796466158700749])
    # opt_2 = scaling(opt_2, -1*np.ones(3), 1*np.ones(3), 1) # Normalize variables between -1 and 1 back to 0 and 1
    # opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_2 = opt_2_unscaled[0] + opt_2_unscaled[2] - 201

    # opt_3 = np.array([0.340000000000000, 0.740000000000000, 0.730000000000000])
    # opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_3 = opt_3_unscaled[0] + opt_3_unscaled[2] - 201

    # print('point #1: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2],g_lin_1))
    # print('point #2: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2],g_lin_2))
    # print('point #3: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2],g_lin_3))
    
    # points = np.vstack((opt_1,opt_2,opt_3))

    # labels = ['$\mathtt{StoMADS-PB}$ candidate solution 1 $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1[0],opt_1[1],opt_1[2]),
    #           '$\mathtt{StoMADS-PB}$ candidate solution 2 $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2[0],opt_2[1],opt_2[2]),
    #           '$\mathtt{NOMAD}$ candidate solution 3 $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3[0],opt_3[1],opt_3[2])]
    
    #===================================================================#
    # R1 opts (new algorithm)
    
    # # Points to plot
    # opt_1 = np.array([0.359651015197352,  0.471324654138228,  0.492508007612736])
    # opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_1 = opt_1_unscaled[0] + opt_1_unscaled[2] - 201

    # opt_2 = np.array([0.362162859400232, 0.726644087036705, 0.439795393281029])
    # opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_2 = opt_2_unscaled[0] + opt_2_unscaled[2] - 201

    # opt_3 = np.array([0.340000000000000, 0.740000000000000, 0.730000000000000])
    # opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_3 = opt_3_unscaled[0] + opt_3_unscaled[2] - 201

    # print('point #1: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2],g_lin_1))
    # print('point #2: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2],g_lin_2))
    # print('point #3: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2],g_lin_3))

    # points = np.vstack((opt_1,opt_2,opt_3))

    # labels = ['$\mathtt{StoMADS-PB}$, sample rate ($p^k$) = 10: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1[0],opt_1[1],opt_1[2]),
    #           '$\mathtt{StoMADS-PB}$ candidate solution 1, sample rate ($p^k$) = 5: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2[0],opt_2[1],opt_2[2]),
    #           '$\mathtt{NOMAD}$ solution $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3[0],opt_3[1],opt_3[2])]
    
    #===================================================================#
    # R0 & R1 opts
    
    # Points to plot
    opt_1 = np.array([0.00007426524509085878, 0.42380805903074963981, 0.02843359294084374031])
    opt_1 = scaling(opt_1, -1*np.ones(3), 1*np.ones(3), 1) # Normalize variables between -1 and 1 back to 0 and 1
    opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)
    g_lin_1 = opt_1_unscaled[0] + opt_1_unscaled[2] - 201

    opt_2 = np.array([0.06624374389648435280, 0.48429003953933713600, 0.14601796466158700749])
    opt_2 = scaling(opt_2, -1*np.ones(3), 1*np.ones(3), 1) # Normalize variables between -1 and 1 back to 0 and 1
    opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)
    g_lin_2 = opt_2_unscaled[0] + opt_2_unscaled[2] - 201

    opt_3 = np.array([0.359651015197352,  0.471324654138228,  0.492508007612736])
    opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)
    g_lin_3 = opt_3_unscaled[0] + opt_3_unscaled[2] - 201

    opt_4 = np.array([0.362162859400232, 0.726644087036705, 0.439795393281029])
    opt_4_unscaled = scaling(opt_4, bounds[:3,0], bounds[:3,1], 2)
    g_lin_4 = opt_4_unscaled[0] + opt_4_unscaled[2] - 201

    opt_5 = np.array([0.340000000000000, 0.740000000000000, 0.730000000000000])
    opt_5_unscaled = scaling(opt_5, bounds[:3,0], bounds[:3,1], 2)
    g_lin_5 = opt_5_unscaled[0] + opt_5_unscaled[2] - 201

    print('point #1: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2],g_lin_1))
    print('point #2: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2],g_lin_2))
    print('point #3: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2],g_lin_3))
    print('point #4: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_4_unscaled[0],opt_4_unscaled[1],opt_4_unscaled[2],g_lin_4))
    print('point #5: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_5_unscaled[0],opt_5_unscaled[1],opt_5_unscaled[2],g_lin_5))

    points = np.vstack((opt_1,opt_2,opt_3,opt_4,opt_5))

    # labels = ['$\mathtt{StoMADS-PB}$ V1, sample rate ($p^k$) = 10: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1[0],opt_1[1],opt_1[2]),
    #           '$\mathtt{StoMADS-PB}$ V1, sample rate ($p^k$) = 10: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2[0],opt_2[1],opt_2[2]),
    #           '$\mathtt{StoMADS-PB}$ V2, sample rate ($p^k$) = 10: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3[0],opt_3[1],opt_3[2]),
    #           '$\mathtt{StoMADS-PB}$ V2, sample rate ($p^k$) = 5: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_4[0],opt_4[1],opt_4[2]),
    #           '$\mathtt{NOMAD}$ solution $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_5[0],opt_5[1],opt_5[2])]

    labels = ['$\mathtt{StoMADS-PB}$ 1: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1[0],opt_1[1],opt_1[2]),
              '$\mathtt{StoMADS-PB}$ 2: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2[0],opt_2[1],opt_2[2]),
              '$\mathtt{StoMADS-PB}$ 3: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3[0],opt_3[1],opt_3[2]),
              '$\mathtt{StoMADS-PB}$ 4: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_4[0],opt_4[1],opt_4[2]),
              '$\mathtt{NOMAD}$: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_5[0],opt_5[1],opt_5[2])]


    #===================================================================#
    # Select points
    
    # # Points to plot
    # opt_1 = np.array([0.00007426524509085878, 0.42380805903074963981, 0.02843359294084374031])
    # opt_1 = scaling(opt_1, -1*np.ones(3), 1*np.ones(3), 1) # Normalize variables between -1 and 1 back to 0 and 1
    # opt_1_unscaled = scaling(opt_1, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_1 = opt_1_unscaled[0] + opt_1_unscaled[2] - 201

    # opt_2 = np.array([0.362162859400232, 0.726644087036705, 0.439795393281029])
    # opt_2_unscaled = scaling(opt_2, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_2 = opt_2_unscaled[0] + opt_2_unscaled[2] - 201

    # opt_3 = np.array([0.340000000000000, 0.740000000000000, 0.730000000000000])
    # opt_3_unscaled = scaling(opt_3, bounds[:3,0], bounds[:3,1], 2)
    # g_lin_3 = opt_3_unscaled[0] + opt_3_unscaled[2] - 201

    # print('point #1: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_1_unscaled[0],opt_1_unscaled[1],opt_1_unscaled[2],g_lin_1))
    # print('point #2: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_2_unscaled[0],opt_2_unscaled[1],opt_2_unscaled[2],g_lin_2))
    # print('point #3: x1 = %f, x2 = %f, x3 = %f, g_lin = %f' %(opt_3_unscaled[0],opt_3_unscaled[1],opt_3_unscaled[2],g_lin_3))

    # points = np.vstack((opt_1,opt_2,opt_3))

    # labels = ['$\mathtt{StoMADS-PB}$ V1, sample rate ($p^k$) = 10: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_1[0],opt_1[1],opt_1[2]),
    #           '$\mathtt{StoMADS-PB}$ V2, sample rate ($p^k$) = 5: $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_2[0],opt_2[1],opt_2[2]),
    #           '$\mathtt{NOMAD}$ solution $\mathbf{x}_{\mathrm{scaled}} = [%.3g ~ %.3g ~ %.3g]^{\mathrm{T}}$' %(opt_3[0],opt_3[1],opt_3[2])]

    #===================================================================#

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


    same_axis = True
    if same_axis:
        fig_nsafety = plt.figure(figsize=(6,5))
    else:
        fig_nsafety = None

    auto_limits = False
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

    if color_mode == 'color':
        hatches = ['/'] * 10
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', ...]
    elif color_mode == 'black_white':
        hatches = ['/','//','x','o','|||']
        colors = ['#FFFFFF'] * 10

    for point,legend_label in zip(points,labels):

        filename = 'MCS_results/DOE_R%i.mat' %(run+1)
        mat = scipy.io.loadmat(filename) # get optitrack data
        data = mat['MCS_runs']
        data = data[:,0::]
        
        # Legend entries
        a = patches.Rectangle((20,20), 20, 20, linewidth=1, edgecolor='k', facecolor=colors[run], fill=True ,hatch=hatches[run])

        handles_lgd += [a]
        labels_lgd += [legend_label]

        label_name = u'Stochastic objective function ($\hat{f}_{\mathbf{P}}(\mathbf{x})$)'
        fun_name = 'nsafety'

        dataXLim_out, dataYLim_out, mean_nsafety, std_nsafety = plot_distribution(data, fun_name, label_name, n_bins, run, 
            fig_swept = fig_nsafety, run_label = legend_label, color = colors[run], 
            hatch_pattern = hatches[run], dataXLim = dataXLim, dataYLim = dataYLim, 
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
        fig_nsafety.savefig('MCS_results/PDF_%s.pdf' %('n_safety'), 
                                format='pdf', dpi=100,bbox_inches='tight')
        plt.show()