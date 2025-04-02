import matplotlib.pyplot as plt

ABS_MIN = 0
ABS_MAX = 1.1
FIGSIZE = (6,5)

def get_broken_figure(bottom_lim=None, top_lim=None):
    if bottom_lim == None:
        fig, (t_ax, b_ax) = plt.subplots(2, 1,
                            sharex=True,
                            gridspec_kw={
                                'height_ratios': [0.05, 1]
                            },
                            figsize=FIGSIZE)
        fig.subplots_adjust(hspace=0.05)
        b_ax.set_ylim(ABS_MIN, top_lim)
        t_ax.set_ylim(top_lim, ABS_MAX)
        # 
        t_ax.set_yticks([1])
        t_ax.set_yticklabels(['1'])

        # t_ax.set_xticks([])
        # t_ax.set_xticklabels([])
        t_ax.spines['bottom'].set_visible(False)
        b_ax.spines['top'].set_visible(False)

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-1, -d), (1, d)],
            markersize=12,  # "length" of cut-line
            linestyle='none',
            color='k',  # ?
            mec='k',  # ?
            mew=1,  # line thickness
            clip_on=False
        )
        # top cut
        t_ax.plot([0, 1], [0, 0], transform=t_ax.transAxes, **kwargs)
        b_ax.plot([0, 1], [1, 1], transform=b_ax.transAxes, **kwargs)

        t_ax.grid()
        return fig, t_ax, b_ax
    if top_lim == None:
        fig, (t_ax, b_ax) = plt.subplots(2, 1,
                            sharex=True,
                            gridspec_kw={
                                'height_ratios': [1, 0.05]
                            },
                            figsize=FIGSIZE)

        fig.subplots_adjust(hspace=0.05)

        t_ax.set_ylim(bottom_lim, ABS_MAX)
        b_ax.set_ylim(ABS_MIN, bottom_lim)
        # limit range of y-axis to the data only

        # remove x-axis line's between the two sub-plots
        t_ax.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
        # m_ax.spines['bottom'].set_visible(False)
        # m_ax.spines['top'].set_visible(False)
        b_ax.spines['top'].set_visible(False)
        # t_ax.set_yticks([1]) # Only show 1 on y-axis
        b_ax.set_yticks([0])  # Only show 0 on y-axis
        b_ax.set_yticklabels(['0'])

        # now draw the cut
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-1, -d), (1, d)],
            markersize=12,  # "length" of cut-line
            linestyle='none',
            color='k',  # ?
            mec='k',  # ?
            mew=1,  # line thickness
            clip_on=False
        )
        # top cut
        t_ax.plot([0, 1], [0, 0], transform=t_ax.transAxes, **kwargs)
        b_ax.plot([0, 1], [1, 1], transform=b_ax.transAxes, **kwargs)
        t_ax.grid()
        # b_ax.grid()
        return fig, t_ax, b_ax
    else:
        fig, (t_ax, m_ax, b_ax) = plt.subplots(3, 1,
                            sharex=True,
                            gridspec_kw={
                                'height_ratios': [0.05, 1, 0.05]
                            },
                            figsize=FIGSIZE)

        fig.subplots_adjust(hspace=0.05)

        t_ax.set_ylim(top_lim, ABS_MAX) 
        m_ax.set_ylim(bottom_lim, top_lim)
        b_ax.set_ylim(ABS_MIN, bottom_lim)
        # limit range of y-axis to the data only

        # remove x-axis line's between the two sub-plots
        t_ax.spines['bottom'].set_visible(False)  # 1st subplot bottom x-axis
        m_ax.spines['bottom'].set_visible(False)
        m_ax.spines['top'].set_visible(False)
        b_ax.spines['top'].set_visible(False)

        t_ax.tick_params(labeltop=False)  # no labels
        m_ax.xaxis.set_ticks_position('none')
        t_ax.set_yticks([1]) # Only show 1 on y-axis
        b_ax.set_yticks([0])  # Only show 0 on y-axis
        b_ax.set_yticklabels(['0'])

        # now draw the cut
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(
            marker=[(-1, -d), (1, d)],
            markersize=12,  # "length" of cut-line
            linestyle='none',
            color='k',  # ?
            mec='k',  # ?
            mew=1,  # line thickness
            clip_on=False
        )
        # top cut
        t_ax.plot([0, 1], [0, 0], transform=t_ax.transAxes, **kwargs)
        # m_ax.plot([0, 1], [1, 1], transform=m_ax.transAxes, **kwargs)
        # bottom cut
        m_ax.plot([0, 1], [0, 0], transform=m_ax.transAxes, **kwargs)
        b_ax.plot([0, 1], [1, 1], transform=b_ax.transAxes, **kwargs)
        m_ax.grid()
        # b_ax.grid()
        return fig, t_ax, m_ax, b_ax