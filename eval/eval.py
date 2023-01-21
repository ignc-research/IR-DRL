import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
import random
from operator import truediv


class qual:
    def __init__(self, pth, planner, single="True"):
        self.trj_pth = pth
        self.planner = planner
        self.evaluations = {}
        self.single = single

    def cuboid_data2(self, o, size=(1, 1, 1)):
        X = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(o)
        return X

    def plotCubeAt2(self, positions, sizes=None, colors=None, **kwargs):
        if not isinstance(colors, (list, np.ndarray)):
            colors = ["C0"] * len(positions)
        if not isinstance(sizes, (list, np.ndarray)):
            sizes = [(1, 1, 1)] * len(positions)
        g = []
        for p, s, c in zip(positions, sizes, colors):
            g.append(self.cuboid_data2(p, size=s))
        return Poly3DCollection(
            np.concatenate(g), facecolors=np.repeat(colors, 6), **kwargs
        )

    def comp_smoothness(self,x,y,z):
        idx = 0
        smoothness = 0 
        while idx < len(x)-2:
            p1 = np.array([x[idx],y[idx],z[idx]])
            p2 = np.array([x[idx+1],y[idx+1],z[idx+1]])
            p3 = np.array([x[idx+2],y[idx+2],z[idx+2]])
            e1 = p2-p1
            e2 = p3-p2
            # np.linalg(e2-e1) 
            smoothness += np.linalg.norm(e2-e1) 
            idx += 1
        # print(self.planner, smoothness)
        return smoothness




    def plot_trj2(self):
        arr = os.listdir(self.trj_pth)
        arr.sort()

        max_el, n_runs = self.count_elems(arr)
        x_av = np.zeros(max_el)
        y_av = np.zeros(max_el)
        z_av = np.zeros(max_el)

        runtime = np.zeros(n_runs)
        tr_len = np.zeros(n_runs)
        ee_v = np.zeros(n_runs)
        smoothness = np.zeros(n_runs)

        # file_id = np.zeros(n_runs)
        # global ax
        if plt_cfg["ql"] == 2:
            global ax
        elif plt_cfg["ql"] == 1:
            ax = plt.axes(projection="3d")

        n = 0
        # print(arr)
        for f in arr:
            if ".txt" in f and not "readme" in f:
                txt_arr = np.loadtxt(self.trj_pth + f)

                ts = txt_arr[:, 0:1].flatten()
                # print(f)
                duration = ts[len(ts) - 1] - ts[0]
                runtime[n] = duration
                # print(duration)

                x = txt_arr[:, 1:2].flatten()
                y = txt_arr[:, 2:3].flatten()
                z = txt_arr[:, 3:4].flatten()
                ev = txt_arr[:, 8:9].flatten()

                # smoothness of path
                # break

                # if 'kuka' in self.trj_pth:
                #     x = x[::-1]
                #     z = z[::-1]
                #     y = y[::-1]

                # elif 'testcase1' in self.trj_pth:
                #     # 1.731480360031127930e-01 5.204795002937316895e-01 2.639703750610351562e-01
                #     x[0] = 0.1731480360031127930
                #     y[0] = 0.5204795002937316895
                #     z[0] = 0.2639703750610351562
                # elif 'testcase3' in self.trj_pth:
                #     # 2.919085323810577393e-01 5.086095929145812988e-01 2.669740319252014160e-01
                #     x[0] = 0.2919085323810577393
                #     y[0] = 0.5086095929145812988
                #     z[0] = 0.2669740319252014160

                # if plt_cfg["ql"] == 1:
                xm = self.unify_trajs(x, max_el)
                ym = self.unify_trajs(y, max_el)
                zm = self.unify_trajs(z, max_el)

                x_av = np.add(x_av, xm)
                y_av = np.add(y_av, ym)
                z_av = np.add(z_av, zm)

                dist_array = (
                    (x[:-1] - x[1:]) ** 2
                    + (y[:-1] - y[1:]) ** 2
                    + (z[:-1] - z[1:]) ** 2
                )

                cuur_smoothness = self.comp_smoothness(x,y,z)
                curr_path_length = np.sum(np.sqrt(dist_array))
                smoothness[n] = cuur_smoothness
                tr_len[n] = curr_path_length
                ee_v[n] = np.mean(ev)

                # file_id[n] = int(f.replace('.txt',''))
                if plt_cfg["ql"] > 0:
                    ax.plot3D(x, y, z, color=planner_stl[self.planner], alpha=0.2)

                n += 1

        x_av /= n_runs
        y_av /= n_runs
        z_av /= n_runs

        # get initial and goal position
        sx = x_av[0]
        sy = y_av[0]
        sz = z_av[0]

        ex = x_av[max_el - 1]
        ey = y_av[max_el - 1]
        ez = z_av[max_el - 1]
        # plot avg trajectory
        if plt_cfg["ql"] > 0:
            ax.plot3D(
                x_av,
                y_av,
                z_av,
                label=self.planner,
                linewidth=3,
                color=planner_stl[self.planner],
            )
        # start and goal marker
        if plt_cfg["ql"] > 0:
            ax.scatter([sx], [sy], [sz], color="b", marker="o", s=150)
            ax.scatter([ex], [ey], [ez], color="r", marker="*", s=150)

            # plot obstacles
            positions = []
            if "testcase1" in self.trj_pth:
                cx = 0
                cy = 0.4
                cz = 0.3
                dx = 0.004
                dy = 0.2
                dz = 0.1
                positions = [(cx - dx / 2, cy - dy / 2, cz - dz / 2)]
                sizes = [(dx, dy, dz)]

                ax.set_xlim([-0.15, 0.15])
                ax.set_ylim([0.375, 0.525])
                ax.set_zlim([0.26, 0.38])
                ax.set_xticks([-0.15, 0, 0.15])
                ax.set_yticks([0.375, 0.45, 0.525])

            elif "testcase2" in self.trj_pth:
                cx = -0.3
                cy = 0.4
                cz = 0.29
                dx = 0.1
                dy = 0.1
                dz = 0.004
                positions = [(cx - dx / 2, cy - dy / 2, cz - dz / 2)]
                sizes = [(dx, dy, dz)]
            elif "testcase3" in self.trj_pth:
                cx = -0.1
                cy = 0.4
                cz = 0.25
                cx2 = 0.1
                cy2 = 0.4
                cz2 = 0.25
                dx = 0.004
                dy = 0.2
                dz = 0.1

                positions = [
                    (cx - dx / 2, cy - dy / 2, cz - dz / 2),
                    (cx2 - dx / 2, cy2 - dy / 2, cz2 - dz / 2),
                ]
                sizes = [(dx, dy, dz), (dx, dy, dz)]
                ax.set_zlim([0.2, 0.33])
                # 2nd goal
                ax.scatter(0, 0.4, 0.25, color="r", marker="*", s=150)

                ax.set_xlim([-0.3, 0.3])
                ax.set_ylim([0.375, 0.525])
                ax.set_zlim([0.2, 0.32])

                # ax.set_yticks([0.375, 0.525])
                # ax.set_yticks([0.375, 0.45, 0.525])

            elif "kukax" in self.trj_pth:
                cx1 = 0
                cy1 = 0.25
                cz1 = 0.6

                cx2 = 0
                cy2 = 0.25
                cz2 = 1.1

                cx3 = 0
                cy3 = 0.25
                cz3 = 1.6

                cx4 = -0.5
                cy4 = 0.25
                cz4 = 1.1

                cx5 = 0
                cy5 = 0.25
                cz5 = 1.1

                cx6 = 0.5
                cy6 = 0.25
                cz6 = 1.1

                dx1 = 1
                dy1 = 0.3
                dz1 = 0.06

                dx2 = 0.06
                dy2 = 0.3
                dz2 = 1

                positions = [
                    (cx1 - dx1 / 2, cy1 - dy1 / 2, cz1 - dz1 / 2),
                    (cx2 - dx1 / 2, cy2 - dy1 / 2, cz2 - dz1 / 2),
                    (cx3 - dx1 / 2, cy3 - dy1 / 2, cz3 - dz1 / 2),
                    (cx4 - dx2 / 2, cy4 - dy2 / 2, cz4 - dz2 / 2),
                    (cx5 - dx2 / 2, cy5 - dy2 / 2, cz5 - dz2 / 2),
                    (cx6 - dx2 / 2, cy6 - dy2 / 2, cz6 - dz2 / 2),
                ]

                grid = []

                sizes = [
                    (dx1, dy1, dz1),
                    (dx1, dy1, dz1),
                    (dx1, dy1, dz1),
                    (dx2, dy2, dz2),
                    (dx2, dy2, dz2),
                    (dx2, dy2, dz2),
                ]

                off = -0.05
                # if 'kuka_3' in self.trj_pth or 'kuka_1' in self.trj_pth:
                #     # off = 0.05
                # # else:
                #     off = -0.05
                # hrz
                ax.plot(
                    [off, off],
                    [0.25 - 1.5, 0.25 + 1.5],
                    [0.6, 0.6],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    [off, off],
                    [0.25 - 1.5, 0.25 + 1.5],
                    [1.1, 1.1],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    [off, off],
                    [0.25 - 1.5, 0.25 + 1.5],
                    [1.6, 1.6],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )

                # ax.plot([0,0],[0.6,0.6] ,[0.25-1.5,0.25+1.5] , color="r")
                # ax.plot([0,0],[1.1,1.1] ,[0.25-1.5,0.25+1.5] , color="r")
                # ax.plot([0,0],[1.6,1.6] ,[0.25-1.5,0.25+1.5] , color="r")

                # vrtc
                ax.plot(
                    [off, off],
                    [0.25 - 1.5, 0.25 - 1.5],
                    [1.6, 0.6],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    [off, off],
                    [0.25 + 1.5, 0.25 + 1.5],
                    [1.6, 0.6],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )
                ax.plot(
                    [off, off],
                    [0.25, 0.25],
                    [1.6, 0.6],
                    color="crimson",
                    linewidth=5,
                    alpha=0.7,
                )

                # ax.plot([0,0],[1.6,0.6] ,[0.25-1.5,0.25-1.5] , color="r")
                # ax.plot([0,0],[1.6,0.6] ,[0.25+1.5,0.25+1.5] , color="r")
                # ax.plot([0,0],[1.6,0.6] ,[0.25,0.25] , color="r")

                # plt.show()
                positions = []
                # ax.set_zlim([0, 2])

            # if plt_cfg["ql"] > 0:
            #     ax.plot3D(
            #         x_av,
            #         y_av,
            #         z_av,
            #         label=self.planner,
            #         linewidth=3,
            #         color=planner_stl[self.planner],
            #     )

            if len(positions) > 0:
                # print(positions)
                colors = []
                for o in positions:
                    colors.append("crimson")
                    # colors.append("grey")
                # ax.set_aspect('auto')
                pc = self.plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
                ax.add_collection3d(pc)

            ax.set_xlabel("x in [m]")
            ax.set_ylabel("y in [m]")
            ax.set_zlabel("z in [m]")

            # if 'kuka' in self.trj_pth:
            #     ax.set_yticks([-0.5, 0])


            # ax.set_xlim([-0.8,0.3])
            # ax.set_ylim([-0.8,0.5])
            # ax.set_zlim([0.4,1.1])

            # ax.view_init(elev=10, azim=230)
            ax.view_init(elev=10, azim=200)
            ax.set_aspect('auto')

        # path_length /= n_runs
        # print(path_length,np.mean(tr_len))
        self.evaluations["smoothness" + self.planner] = smoothness
        self.evaluations["tr_len_" + self.planner] = tr_len
        self.evaluations["eev_" + self.planner] = ee_v
        self.evaluations["runtime_" + self.planner] = runtime
        self.evaluations["tr_av_" + self.planner] = [x_av, y_av, z_av]
        return self.evaluations


    def plot_trj(self):
        csv_files = os.listdir(self.trj_pth)
        # arr.sort()
        print(csv_files)
        max_el, n_runs = self.count_elems(arr)
        x_av = np.zeros(max_el)
        y_av = np.zeros(max_el)
        z_av = np.zeros(max_el)

        runtime = np.zeros(n_runs)
        tr_len = np.zeros(n_runs)
        ee_v = np.zeros(n_runs)
        smoothness = np.zeros(n_runs)

        # file_id = np.zeros(n_runs)
        # global ax
        if plt_cfg["ql"] == 2:
            global ax
        elif plt_cfg["ql"] == 1:
            ax = plt.axes(projection="3d")

        n = 0
        # print(arr)
        for f in arr:
            if ".txt" in f and not "readme" in f:
                txt_arr = np.loadtxt(self.trj_pth + f)

                ts = txt_arr[:, 0:1].flatten()
                # print(f)
                duration = ts[len(ts) - 1] - ts[0]
                runtime[n] = duration
                # print(duration)

                x = txt_arr[:, 1:2].flatten()
                y = txt_arr[:, 2:3].flatten()
                z = txt_arr[:, 3:4].flatten()
                ev = txt_arr[:, 8:9].flatten()


                xm = self.unify_trajs(x, max_el)
                ym = self.unify_trajs(y, max_el)
                zm = self.unify_trajs(z, max_el)

                x_av = np.add(x_av, xm)
                y_av = np.add(y_av, ym)
                z_av = np.add(z_av, zm)

                dist_array = (
                    (x[:-1] - x[1:]) ** 2
                    + (y[:-1] - y[1:]) ** 2
                    + (z[:-1] - z[1:]) ** 2
                )

                cuur_smoothness = self.comp_smoothness(x,y,z)
                curr_path_length = np.sum(np.sqrt(dist_array))
                smoothness[n] = cuur_smoothness
                tr_len[n] = curr_path_length
                ee_v[n] = np.mean(ev)

                # file_id[n] = int(f.replace('.txt',''))
                if plt_cfg["ql"] > 0:
                    ax.plot3D(x, y, z, color=planner_stl[self.planner], alpha=0.2)

                n += 1

        x_av /= n_runs
        y_av /= n_runs
        z_av /= n_runs

        # get initial and goal position
        sx = x_av[0]
        sy = y_av[0]
        sz = z_av[0]

        ex = x_av[max_el - 1]
        ey = y_av[max_el - 1]
        ez = z_av[max_el - 1]

        # plot avg trajectory
        if plt_cfg["ql"] > 0:
            ax.plot3D(
                x_av,
                y_av,
                z_av,
                label=self.planner,
                linewidth=3,
                color=planner_stl[self.planner],
            )
        # start and goal marker
        if plt_cfg["ql"] > 0:
            ax.scatter([sx], [sy], [sz], color="b", marker="o", s=150)
            ax.scatter([ex], [ey], [ez], color="r", marker="*", s=150)


            ax.set_xlabel("x in [m]")
            ax.set_ylabel("y in [m]")
            ax.set_zlabel("z in [m]")

            # if 'kuka' in self.trj_pth:
            #     ax.set_yticks([-0.5, 0])


            # ax.set_xlim([-0.8,0.3])
            # ax.set_ylim([-0.8,0.5])
            # ax.set_zlim([0.4,1.1])

            # ax.view_init(elev=10, azim=230)
            ax.view_init(elev=10, azim=200)
            ax.set_aspect('auto')

        # path_length /= n_runs
        # print(path_length,np.mean(tr_len))
        self.evaluations["smoothness" + self.planner] = smoothness
        self.evaluations["tr_len_" + self.planner] = tr_len
        self.evaluations["eev_" + self.planner] = ee_v
        self.evaluations["runtime_" + self.planner] = runtime
        self.evaluations["tr_av_" + self.planner] = [x_av, y_av, z_av]
        return self.evaluations

    # returns the minimum amount of steps in each file
    def count_elems(self, arr):
        elements = np.zeros(30)
        n = 0
        for f in arr:
            if ".txt" in f and not "readme" in f:
                txt_arr = np.loadtxt(self.trj_pth + f)
                x = txt_arr[:, 0:1]
                elements[n] = len(x)
                # if 'RRT' in self.trj_pth:
                #     print(self.trj_pth,len(x))
                n += 1

        return int(min(elements)), n

    # adjust dimensions of trj arrays
    def unify_trajs(self, trj, max_el):
        modified_trj = trj
        diff = len(trj) - max_el
        n = 0
        while n < diff:
            #    modified_trj.pop(n+1)

            if (len(modified_trj) <= n + 1):  # if diff is too big, then delete earlier values
                modified_trj = np.delete(modified_trj, random.randint(1, 3))
            else:
                modified_trj = np.delete(modified_trj, n + 1)
            n += 1

        return modified_trj

def makeplot_qual(eval_path, robot):

    experiments = {}
    if plt_cfg["ql"] == 2:
        global ax
        plt.figure(robot)
        ax = plt.axes(projection="3d")
    # ax = plt.axes(projection="3d")
    for p in os.listdir(eval_path):
        planner = p
        # print(p, os.path.isdir(eval_path+'/'+p))
        title = robot + "_" + planner
        filepath = eval_path + planner + "/"
        if os.path.isdir(filepath) and planner in plt_cfg["planner"]:
            if plt_cfg["ql"] == 1:
                plt.figure(title)
            # create class obj
            quali = qual(filepath, planner, True)
            # call plot func
            evals = quali.plot_trj()
            dist = evals["tr_len_" + planner]
            dur = evals["runtime_" + planner]
            ev = evals["eev_" + planner]
            smooth = evals["smoothness" + planner]
            experiments[title + ": " + "tr_len"] = dist
            experiments[title + ": " + "dur"] = dur
            experiments[title + ": " + "eev"] = ev
            experiments[title + ": " + "smooth"] = smooth
            if plt_cfg["ql"] == 1:
                plt.legend()
                plt.savefig("quali/" + title + ".png", bbox_inches="tight")
            elif plt_cfg["ql"] == 2:
                plt.legend()
                plt.savefig("quali/" + robot + ".png", bbox_inches="tight")
            # makeplot_line(dur, planner, "duration")
            # makeplot_line(dist, planner, "tr_len")

            if plt_cfg["qt"] == 1:
                makeplot_bar(dur, planner, "duration_" + title)
                makeplot_bar(dist, planner, "path length_" + title)

    # print(bar_data)
    if plt_cfg["qt"] == 2:
        makeplot_bar(experiments, "", robot)
    

    return experiments


def makeplot_bar(experiments, planner, ttl):
    if plt_cfg["qt"] == 1:
        N = len(experiments)
        x = np.linspace(1, N, N, endpoint=True)
        plt.figure(ttl)
        plt.xlabel("Test run")
        if "dur" in ttl:
            plt.ylabel("Execution Time in [s]")
        else:
            plt.ylabel("Path Length in [m]")

        plt.bar(x, experiments, color=planner_stl[planner])
        plt.axhline(y=np.mean(experiments), color=planner_stl[planner], linewidth=2)
    else:
        # plt.title(ttl)
        data_dur = {}
        data_len = {}
        for key in experiments:
            # print(key)
            if "dur" in key:
                data_dur[key] = experiments[key]
            elif "len" in key:
                data_len[key] = experiments[key]

        fig, ax = plt.subplots()
        ax.set_title(ttl)
        # print(mean_dur)
        bar_plot(ax, data_dur, total_width=0.8, single_width=0.9)
        fig, ax = plt.subplots()
        ax.set_title(ttl)
        bar_plot(ax, data_len, total_width=0.8, single_width=0.9)
    plt.savefig("quanti/" + ttl + ".png", bbox_inches="tight")


def makeplot_line(arr, planner, ttl):
    N = len(arr)
    x = np.linspace(1, N, N, endpoint=True)

    plt.figure(ttl)
    plt.plot(
        x,
        arr,
        "o:",
        color=planner_stl[planner],
        label=planner + ": t_av = " + str(round(np.mean(arr), 2)),
    )

    # plt.ylim((0.7,0.9))
    plt.xlabel("run")
    plt.ylabel("execution time in [s]")
    plt.legend()


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, ns=""):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # print(colors)
    # Number of bars per group
    n_bars = len(data)
    print(data)
    if n_bars > 0:
        # The width of a single bar
        # print(data)
        bar_width = total_width / n_bars

        # List containing handles for the drawn bars, used for the legend
        bars = []

        # Iterate over all data
        for i, (name, values) in enumerate(data.items()):
            # The offset in x direction of that bar
            if "kuka" in ns:
                x_offset = 4 + (i - n_bars / 2) * bar_width + bar_width / 2
            else:
                x_offset = 1 + (i - n_bars / 2) * bar_width + bar_width / 2

            planner = ""

            # Draw a bar for every value of that type
            for x, y in enumerate(values):
            #     # print(name)
            #     if "DRL" in name and not "JV" in name:
            #         planner = "DRL"
            #     elif "JV" in name:
            #         planner = "DRL-JV"
            #     elif "NC-RRT" in name:
            #         planner = "NC-RRT"
            #     elif "RRT" in name:
            #         planner = "RRT"

            #     # planner = name
            #     if planner == "":
            #         clr = colors[i % len(colors)]
            #     else:
            #         clr = planner_stl[planner]
                clr = planner_stl[name]

                bar = ax.bar(
                    x + x_offset,
                    y,
                    width=bar_width * single_width,
                    color=clr,
                )

            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])

        # Draw legend if we need
        if legend:
            ax.legend(bars, data.keys(), loc=0)
        for bars in ax.containers:
            # print(type(bars[0]),bars)
            # b = round(bars,1)
            ax.bar_label(bars, fmt="%.2f", fontweight="bold")
    else:
        print("No data to plot: ", data)


def bar_av(experiments, ttl, plnr):

    planner = plnr.split(",")

    data_dur = {}
    mean_dur = {}
    rel_dur = {}

    data_len = {}
    mean_len = {}
    rel_len = {}

    data_vel = {}
    mean_vel = {}
    rel_vel = {}

    data_smooth = {}
    mean_smooth = {}
    rel_smooth = {}

    # format dict according bar function
    for exp in experiments:
        for key in exp:
            if "dur" in key:
                data_dur[key] = exp[key]
                for p in planner:
                    if "_" + p in key:
                        if p in mean_dur:
                            mean_dur[p].append(np.mean(exp[key]))
                        else:
                            # if 'NC-RRT' in p:
                            #     mean_dur[p].append(np.mean(exp[key])+0.2)
                            mean_dur[p] = [np.mean(exp[key])]
            elif "len" in key:
                data_len[key] = exp[key]
                for p in planner:
                    if "_" + p in key:
                        if p in mean_len:
                            mean_len[p].append(np.mean(exp[key]))
                        else:
                            mean_len[p] = [np.mean(exp[key])]
            elif "eev" in key:
                data_vel[key] = exp[key]
                for p in planner:
                    if "_" + p in key:
                        if p in mean_vel:
                            mean_vel[p].append(np.mean(exp[key]))
                        else:
                            mean_vel[p] = [np.mean(exp[key])]
            elif "smooth" in key:
                data_smooth[key] = exp[key]
                for p in planner:
                    if "_" + p in key:
                        if p in mean_smooth:
                            mean_smooth[p].append(np.mean(exp[key]))
                        else:
                            mean_smooth[p] = [np.mean(exp[key])]

    # bar_3d(data_len)

    for p in mean_dur:
        if not "DRL" in p:
            # diif_dur = [element1 - element2 for (element1, element2) in zip(mean_dur["DRL"], mean_dur[p])]
            # diif_len = [element1 - element2 for (element1, element2) in zip(mean_len["DRL"], mean_len[p])]
            # diif_vel = [element1 - element2 for (element1, element2) in zip(mean_vel["DRL"], mean_vel[p])]

            # rel_len[p] = list(map(truediv, diif_dur, mean_len["DRL"]))
            # rel_dur[p] = list(map(truediv, diif_len, mean_dur["DRL"]))
            # rel_vel[p] = list(map(truediv, diif_vel, mean_vel["DRL"]))

            rel_len[p] = list(map(truediv, mean_len[p], mean_len["DRL"]))
            rel_dur[p] = list(map(truediv, mean_dur[p], mean_dur["DRL"]))
            rel_vel[p] = list(map(truediv, mean_vel[p], mean_vel["DRL"]))

    # print(mean_len)
    # print(ttl)
    # print('dur',mean_dur)
    fig, ax = plt.subplots()
    # ax.set_title("dur")
    bar_plot(ax, mean_dur, total_width=0.7, single_width=0.8, ns=ttl)
    plt.xlabel("Experiment")
    plt.ylabel("Average Execution time in [s]", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("quanti/" + ttl + "_dur.png", bbox_inches="tight")

    # rel - enable for relative comparison
    # fig, ax = plt.subplots()
    # # ax.set_title("dur")
    # bar_plot(ax, rel_dur, total_width=0.7, single_width=0.8, ns=ttl)
    # plt.xlabel("Experiment")
    # plt.ylabel("Rel Execution time", fontweight="bold")
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.savefig("quanti/" + ttl + "_rel_dur.png", bbox_inches="tight")

    # print('len',mean_len)
    fig, ax = plt.subplots()
    # ax.set_title("len")
    bar_plot(ax, mean_len, total_width=0.7, single_width=0.8, ns=ttl)
    plt.xlabel("Experiment")
    plt.ylabel("Average path length in [m]", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("quanti/" + ttl + "_len.png", bbox_inches="tight")

    # rel - enable for relative comparison
    # fig, ax = plt.subplots()
    # # ax.set_title("len")
    # bar_plot(ax, rel_len, total_width=0.7, single_width=0.8, ns=ttl)
    # plt.xlabel("Experiment")
    # plt.ylabel("Rel path length", fontweight="bold")
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.savefig("quanti/" + ttl + "_rel_len.png", bbox_inches="tight")

    # print('vel',mean_vel)
    fig, ax = plt.subplots()
    # ax.set_title("len")
    bar_plot(ax, mean_vel, total_width=0.7, single_width=0.8, ns=ttl)
    plt.xlabel("Experiment")
    plt.ylabel("Average velocity in [m/s]", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("quanti/" + ttl + "_vel.png", bbox_inches="tight")

    # rel - enable for relative comparison
    # fig, ax = plt.subplots()
    # # ax.set_title("len")
    # bar_plot(ax, rel_vel, total_width=0.7, single_width=0.8, ns=ttl)
    # plt.xlabel("Experiment")
    # plt.ylabel("Rel velocity", fontweight="bold")
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.savefig("quanti/" + ttl + "_rel_vel.png", bbox_inches="tight")
    # # return

    fig, ax = plt.subplots()
    # ax.set_title("dur")
    bar_plot(ax, mean_smooth, total_width=0.7, single_width=0.8, ns=ttl)
    plt.xlabel("Experiment")
    plt.ylabel("Average Smoothness", fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("quanti/" + ttl + "_smooth.png", bbox_inches="tight")

def bar_3d(experiments):
    # Fixing random state for reproducibility
    planner = plt_cfg["planner"].split(',')
    data = {}
    for key in experiments:
        print(key)
    #     for p in planner:
    #         if 
    #         data[key] = 
    np.random.seed(19680801)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ['r', 'g', 'b', 'y']
    yticks = [4, 3, 2, 1]
    for c, k in zip(colors, yticks):
        # Generate the random data for the y=k 'layer'.
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array with the same length as
        # xs and ys. To demonstrate this, we color the first bar of each set cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'

        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        # print(xs)
        # print(ys)
        ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # On the y axis let's only label the discrete values that we have data for.
    ax.set_yticks(yticks)



if __name__ == "__main__":
    experiments = {}
    planner_stl = {}

    planner_stl["DRL"] = "tab:blue"
    planner_stl["DRL-JV"] = "tab:green"
    planner_stl["RRT"] = "tab:orange"
    planner_stl["NC-RRT"] = "tab:purple"
    planner_stl["RRTs"] = "tab:red"
    planner_stl["DRL-AmirV9"] = "tab:gray"

    plt_cfg = {}
    plt_cfg["ql"] = 2
    plt_cfg["qt"] = 0
    # plt_cfg["planner"] = "DRL,DRL-JV,RRT,NC-RRT,DRL-AmirV9"
    # plt_cfg["planner"] = "NC-RRT,DRL,DRL-AmirV9"
    plt_cfg["planner"] = "DRL"

    # # Ur 5 ----------------------------------------------------------
    # ur5_1 = makeplot_qual("../ur5/trajectory/testcase1/", "ur5_1")
    # ur5_2 = makeplot_qual("../ur5/trajectory/testcase2/", "ur5_2")
    # ur5_3 = makeplot_qual("../ur5/trajectory/testcase3/", "ur5_3")

    ur5_1 = makeplot_qual("../ur5/trajectory/table_exp1/", "ur5_new")


    # # Kuka ----------------------------------------------------------
    # plt_cfg["planner"] = "DRL,RRT,NC-RRT"
    # kuka_1 = makeplot_qual("../kuka/trajectory/lowerleft/", "kuka_1")
    # kuka_2 = makeplot_qual("../kuka/trajectory/lowerright/", "kuka_2")
    # kuka_3 = makeplot_qual("../kuka/trajectory/upperleft/", "kuka_3")
    # kuka_4 = makeplot_qual("../kuka/trajectory/upperright/", "kuka_4")

    # # average bar plots ---------------------------------------------
    # bar_av([ur5_1, ur5_2, ur5_3], "ur5", "DRL,RRT,NC-RRT,DRL-JV")
    # bar_av([ur5_1], "ur5","DRL,NC-RRT,DRL-JV,DRL-AmirV9")
    # bar_av([ur5_1], "ur5", "DRL,DRL-AmirV9")
    # bar_av([kuka_1, kuka_2, kuka_3, kuka_4], "kuka", "DRL,RRT,NC-RRT")

    # 3d bar plots --------------------------------------------------
    # bar_3d() 
    plt.show()
