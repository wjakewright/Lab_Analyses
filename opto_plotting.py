import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('ticks')


def plot_session_activity(dFoF, itis, zscore=False, figsize=(7,8), title='default'):
    '''Function to plot the activity of each ROI for the entire imaging session.
        Indicates the locations of each stimulation.'''
    if title == 'default':
        title = 'Session Activity'
    else:
        title = title + ' Session Activity'
    plt.figure(figsize=figsize)
    for i, col in enumerate(dFoF.columns):
        x = np.linspace(0,len(dFoF[col])/30,len(dFoF[col])) # Will be in units time(s)
        plt.plot(x,dFoF[col] + i*5, label=col, linewidth=0.5)
    
    for iti in itis:
        plt.axvspan(iti[0]/30,iti[1]/30, alpha=0.3, color='red')
    
    plt.tick_params(axis='both', which='both', direction='in')
    plt.xlabel('Time(s)')
    if zscore is True:
        plt.ylabel('z-score')
    else:
        plt.ylabel(r'$\Delta$F/F')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.1,1.05))
    plt.tight_layout()

def plot_each_event(roi_stim_epochs, ROIs, figsize=(7,8), title='default'):
    '''Function to plot the activity of each ROI around each stimulation event.'''
    if title == 'default':
        title = 'Time Locked Activity'
    else:
        title = title + ' Time Locked Activity'
    tot = len(roi_stim_epochs.keys())
    col_num = 1
    row_num = tot//col_num
    row_num += tot%col_num
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=1)
    fig.suptitle(title)

    for count, (key,value) in enumerate(roi_stim_epochs.items()):
        ax = fig.add_subplot(row_num,col_num,count+1)
        win_len = np.shape(value)[0]
        x = np.linspace(0,win_len,win_len) # Will be in units time(s)
        iti_shade = np.array([60,90])
        for col in range(len(value[0,:])):
            if col == 0:
                ax.plot(x,value[:,col],color='mediumblue')
                ax.axvspan(iti_shade[0],iti_shade[1],color='red', alpha=0.1)
            else:
                pad = len(x) + 1
                x = x + pad
                iti_shade = iti_shade + pad
                ax.plot(x,value[:,col],color='mediumblue')
                ax.axvspan(iti_shade[0],iti_shade[1],color='red', alpha=0.1)
        ax.set_title(ROIs[count],fontsize=10)
        ax.tick_params(axis='both',which='both',direction='in',length=4)
    fig.tight_layout()


def plot_mean_sem(roi_mean_sems, new_window, ROIs, figsize=(10,10), col_num=4, main_title='default'):
    if main_title == 'default':
        main_title = 'Mean Opto Activity'
    else:
        main_title = main_title + ' Mean Opto Activity'
    tot = len(roi_mean_sems)
    col_num = col_num
    row_num = tot//col_num
    row_num += tot%col_num
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(main_title)
    
    # Get max amplitude in order to set ylim
    ms = []
    for key,value in roi_mean_sems.items():
        ms.append(np.max(value[0]+value[1]))
    ymax = max(ms)
    
    count = 1
    for key, value in roi_mean_sems.items():
        x = np.linspace(new_window[0],new_window[1],len(value[0]))
        ax = fig.add_subplot(row_num,col_num,count)
        ax.plot(x,value[0],color='mediumblue')
        ax.fill_between(x,value[0]-value[1],value[0]+value[1],
                            color='mediumblue', alpha=0.2)
        ax.axvspan(0,1, alpha=0.1, color='red')
        plt.xticks(ticks = [new_window[0],0,new_window[1]],
                    labels = [new_window[0],0,new_window[1]])
        ax.set_title(ROIs[count-1],fontsize=10)
        ax.tick_params(axis='both', which='both', direction='in',length=4)
        plt.ylim(top=ymax+(ymax*0.1))
        count +=1
    fig.tight_layout()

def plot_opto_heatmap(roi_mean_sems, zscore, sampling_rate, figsize=(4,5), main_title='default', cmap=None):
    '''Function to plot heatmap of avg ROI activity'''

    if main_title == 'default':
        main_title = 'Mean Opto Activity Heatmap'
    else:
        main_title = main_title + ' Mean Opto Activity Heatmap'
    # Custom color map for the heatmap
    d_map = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                                        [(0.0, 'mediumblue'),
                                                            (0.3, 'royalblue'),
                                                            (0.5, 'white'),
                                                            (0.7, 'tomato'),
                                                            (1.0, 'red')],N=1000)
    # Put data in a dataframe to make it easier to plot
    df = pd.DataFrame()
    for key, value in roi_mean_sems.items():
        df[key] = value[0]
    df_t = df.T
    # Plot the heatmap
    if cmap is None:
        cmap=d_map
    else:
        pass
    fig = plt.figure(figsize=figsize)
    fig.suptitle(main_title)
    if zscore is True:
        ax = sns.heatmap(df_t,cmap=cmap,center=0,vmax=2,vmin=-2,
                        cbar_kws={'label':'z-score','orientation':'vertical',
                                    'ticks':(-2,0,2)},yticklabels=False)
    else:
        ax = sns.heatmap(df_t,cmap='inferno',vmax=5,vmin=-0.5,
                        cbar_kws={'label':r'$\Delta$F/F','orientation':'vertical',
                                    'ticks':(-0.5,0,5)},yticklabels=False) 
    
    plt.xticks(ticks = [0,(sampling_rate),(sampling_rate*2),
                        (sampling_rate*3),(sampling_rate*4),
                        (sampling_rate*5)],
                labels = [-2,1,0,1,2,3], rotation=0)
    ax.axvline(x=(sampling_rate*2),ymin=0,ymax=1,color='black',linestyle='--')
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2.5')
    
    fig.tight_layout()

def plot_shuff_distribution(sig_results, ROIs, figsize=(10,10), col_num=4, main_title="default"):
    '''Function to plot the shuffle distribution against the real data for 
        each ROI'''
    
    if main_title == 'default':
        main_title = "Shuffle Distributions"
    else:
        main_title = main_title + ' Shuffle Distributions'
    tot = len(sig_results.keys())
    col_num = col_num
    row_num = tot//col_num
    row_num += tot%col_num
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(main_title)
    
    count = 1
    for key, value in sig_results.items():
        ax = fig.add_subplot(row_num, col_num, count)
        ax.hist(value['shuff_diff'],color='mediumblue',alpha=0.7,
                linewidth=0.5)
        ax.axvline(x=value['real_diff'], color='red',linewidth=2.5)
        ax.axvline(x=value['bounds'][0], color='black',linewidth=1,linestyle='--')
        ax.axvline(x=value['bounds'][1], color='black',linewidth=1,linestyle='--')
        ax.set_title(ROIs[count-1],fontsize=10)
        ax.tick_params(axis='both', which='both', direction='in', length=4)
        
        count += 1
    
    fig.tight_layout()

def plot_power_curve(ps, diffs, sems, scatter, percent_sig, zscore):
    '''Function to plot mean activity and percent significant for different optostimulation
        power sessions.'''
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    p = list(range(len(ps)))
    powers = [str(x) for x in ps]
    ax1.errorbar(p,diffs,yerr=sems,color='red',
                    marker='o',markerfacecolor='red',ecolor='red')
    sns.swarmplot(data=scatter,color='red',size=4,alpha=0.2)
    ax1.axhline(y=0,color='black',linestyle='--',linewidth=1)
    ax1.set_title('Mean Change in Activity',fontsize = 12)
    if zscore is True:
        ylab = 'z-scored $\Delta$F/F'
    else:
        ylab = '$\Delta$F/F'
    ax1.set_ylabel(ylab)
    ax1.set_xticklabels(labels=powers)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(p,percent_sig,color='red',marker='o',markerfacecolor='red')
    ax2.set_title('Percent Significant',fontsize=12)
    ax2.set_ylabel('Percentage of Neurons')
    ax2.set_xticks(p)
    ax2.set_xticklabels(labels=powers)
    plt.ylim(bottom=0)
    
    fig.add_subplot(111,frame_on=False)
    plt.tick_params(labelcolor='none',bottom=False,left=False)
    plt.xlabel('Power (mW)',labelpad=15)
    fig.tight_layout()
    
