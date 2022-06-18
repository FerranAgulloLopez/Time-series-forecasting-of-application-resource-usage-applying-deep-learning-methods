import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 45})
logging.getLogger('matplotlib.font_manager').disabled = True


def plot_correlation_matrix(visualize, matrix, labels, path):
    size = 1.3 * matrix.shape[0]
    # size = 4 * matrix.shape[0]
    fig, ax = plt.subplots(figsize=(size, size))
    df = pd.DataFrame(data=matrix, index=labels, columns=labels)
    sns.heatmap(df, annot=True, xticklabels=list(range(len(labels))), yticklabels=list(range(len(labels))))
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def plot_bar(visualize, filename, title, x_label, y_label, **bar_args):
    plt.subplots(figsize=(20, 10))
    plt.bar(**bar_args)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    if visualize:
        plt.show()


def compare_multiple_lines(visualize, lines, y_label, x_label, title, path, ylim=None, linewidth=3):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, label = line
        ax.plot(x, y, label=label, linewidth=linewidth)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax.legend(loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def compare_multiple_lines_color(visualize, lines, y_label, x_label, title, path, ylim=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, color, linewidth, label = line
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # TODO refactor to dynamic
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='#4f7cac', label='prediction'))
    handles.append(mpatches.Patch(color='#371e30', label='prediction init position'))
    ax.legend(handles=handles, loc='best', shadow=True)
    plt.savefig(path)
    if visualize:
        plt.show()


def compare_multiple_lines_points_color(visualize, lines, points, y_label, x_label, title, path, ylim=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, color, linewidth, label = line
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    for point in points:
        y, x = point
        ax.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # TODO refactor to dynamic
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='#4f7cac', label='prediction'))
    handles.append(mpatches.Patch(color='#371e30', label='prediction init position'))
    ax.legend(handles=handles, loc='best', shadow=True)
    plt.savefig(path)
    if visualize:
        plt.show()


def compare_multiple_lines_points_color_2(visualize, lines, points, y_label, x_label, title, path, ylim=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, color, linewidth, label = line
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    for point in points:
        y, x = point
        ax.plot(x, y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # TODO refactor to dynamic
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc='best', shadow=True)
    plt.savefig(path)
    if visualize:
        plt.show()


def compare_multiple_lines_points_color_extended(visualize, lines, points, y_label, x_label, title, path, ylim=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, color, linewidth, label = line
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    for point in points:
        y, x, color = point
        ax.plot(x, y, marker="o", markersize=15, markeredgecolor=color, markerfacecolor=color)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # TODO refactor to dynamic
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='red', label='chosen samples'))
    ax.legend(handles=handles, loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def compare_multiple_lines_rectangles_color(visualize, lines, rectangles, y_label, x_label, title, path, ylim=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    for line in lines:
        y, x, color, linewidth, label = line
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    for rectangle in rectangles:
        x, y, width, height, angle, args = rectangle
        ax.add_patch(mpatches.Rectangle((x, y), width, height, angle, **args))
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # TODO refactor to dynamic
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color=(1, 0, 0, 0.2), label='prediction'))
    ax.legend(handles=handles, loc='best', shadow=True)
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def compare_multiple_lines_color_matrix(visualize, charts, ncols, title, x_label, path):  # TODO refactor
    nrows = len(charts) // ncols  # TODO refactor when not perfect match
    index = 0
    if nrows > 1:
        raise NotImplementedError()
    else:
        if ncols > 1:
            fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*5))
            for col in range(ncols):
                lines, y_label, subtitle = charts[index]
                for line in lines:
                    y, x, color, linewidth, label = line
                    axs[col].plot(x, y, color=color, linewidth=linewidth, label=label)
                axs[col].legend(loc='best', shadow=True)
                axs[col].set_xlabel(x_label)
                if col == 0:
                    axs[col].set_ylabel(y_label)
                axs[col].set_title(subtitle)
                index += 1
        else:
            raise NotImplementedError()

    plt.suptitle(title)
    plt.savefig(path)
    if visualize:
        plt.show()


def compare_multiple_lines_matrix(visualize, charts, title, x_label, path, ncols=5, ylim=None, legend=True):  # TODO refactor
    nrows = len(charts) // ncols  # TODO refactor when not perfect match
    index = 0
    if nrows > 1:
        if ncols > 1:
            fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 20, ncols * 2))
            for row in range(nrows):
                for col in range(ncols):
                    lines, y_label, subtitle = charts[index]
                    for line in lines:
                        y, x, label = line
                        axs[row, col].plot(x, y, label=label)
                    axs[row, col].legend(loc='best', shadow=True)
                    if row == (nrows - 1):
                        axs[row, col].set_xlabel(x_label)
                    axs[row, col].set_ylabel(y_label)
                    if row == 0:
                        axs[row, col].set_title(subtitle)
                    index += 1
        else:
            fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 5, 20))
            for row in range(nrows):
                lines, y_label, subtitle = charts[index]
                for line in lines:
                    y, x, label = line
                    axs[row].plot(x, y, label=label)
                    # axs[row].set_ylim([0, 10])
                axs[row].set_xlabel(x_label)
                axs[row].set_ylabel(y_label)
                if ylim:
                    axs[row].set_ylim(ylim)
                if row == 0:
                    axs[row].set_title(subtitle)
                if legend:
                    axs[row].legend(loc='best', shadow=True)
                axs[row].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                index += 1
    else:
        if ncols > 1:
            fig, axs = plt.subplots(nrows, ncols, figsize=(40, 10))
            for col in range(ncols):
                lines, y_label, subtitle = charts[index]
                for line in lines:
                    y, x, label = line
                    axs[col].plot(x, y, label=label)
                axs[col].legend(loc='best', shadow=True)
                axs[col].set_xlabel(x_label)
                if ylim:
                    axs[col].ylim(ylim)
                if col == 0:
                    axs[col].set_ylabel(y_label)
                axs[col].set_title(subtitle)
                index += 1
        else:
            fig, axs = plt.subplots(nrows, ncols, figsize=(40, 20))
            lines, y_label = charts[index]
            for line in lines:
                y, x, label = line
                axs.plot(x, y, label=label)
            axs.set_xlabel(x_label)
            axs.set_ylabel(y_label)
            axs.legend(loc='best', shadow=True)

    # plt.suptitle(title)
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def compare_multiple_lines_matrix_specific(visualize, columns_charts, title, y_label, x_label, nrows, path):  # TODO refactor
    ncols = len(columns_charts)

    fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 20, 10))
    for col, column_charts in enumerate(columns_charts):
        for row, chart in enumerate(column_charts):
            lines, y_label, subtitle = chart
            for line in lines:
                y, x, label = line
                axs[row, col].plot(x, y, label=label)
            axs[row, col].legend(loc='best', shadow=True)
            if row == (nrows - 1):
                axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_label)
            if row == 0:
                axs[row, col].set_title(subtitle)
    plt.suptitle(title)
    plt.savefig(path, bbox_inches='tight')
    if visualize:
        plt.show()


def plot_hist(visualize, filename, title, x_label, y_label, **hist_args):
    plt.subplots(figsize=(20, 10))
    plt.hist(**hist_args)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    if visualize:
        plt.show()
