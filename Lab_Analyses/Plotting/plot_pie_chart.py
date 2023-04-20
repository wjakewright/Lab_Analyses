import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("ticks")


def plot_pie_chart(
    data_dict,
    title=None,
    figsize=(5, 5),
    colors="mediumblue",
    alpha=0.7,
    edgecolor="white",
    txt_color="white",
    txt_size=10,
    legend=None,
    donut=0.6,
    linewidth=1.5,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot a pie chart
        
        INPUT PARAMETERS
            data - np.array or list of value counts
            
            title - str specifying the title of the figure
            
            colors - str or list of str to specify the colors of the wedges
            
            alpha - float specifying the color alpha
            
            edgecolor - str specifying the color of the edges of the wedges
            
            txt_color - str specifying the color the labels

            txt_size - int specifying the lable font size
            
            legend - str specifying where to put the legend. Accepts 'top', 'right'
                    and None for no legend
            
            donut - float specifying how long the wedges go towards the center
            
            linewidth - float specifying the thickness of the edges
            
            ax - axis object you wish the data to be plotted on. Usefull for subplotting
            
            save - boolean specifying if you wish to save the figure or not
            
            save_path - str specifying the path of where to save the figure
    """
    # Check if axis was passed
    if ax is None:
        fig, ax = plt.subplot(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

    # Sort data
    labels = list(data_dict.keys())
    data = list(data.values())

    # Sort out the colors
    if type(colors) == str:
        colors = [colors for i in range(len(data))]
    
    # Set the title
    ax.set_title(title)

    # Set up some properties
    wedgeprops = {"width":  donut, "alpha": alpha, "linewidth": linewidth, "edgecolor": edgecolor}
    textprops = {"color": txt_color, "fontsize": txt_size, "weight": "bold"}

    # Plot the data
    wedges, _, _ = ax.pie(
        data,
        wedgeprops=wedgeprops,
        textprops=textprops,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.7,
        colors=colors,
    )

    # Add legend if specified
    if legend == "top":
        ax.legend(
            wedges,
            labels,
            loc="upper center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, 1.05),
            frameon=False,
        )
    elif legend == "right":
        ax.legend(
            wedges,
            labels,
            loc="center right",
            ncol=1,
            bbox_to_anchor=(1, 0, 0.5, 1),
            frameon=False,
        )
    
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")

