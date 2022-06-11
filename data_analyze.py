def describe():
    # import pandas as pd
    # train_data = pd.read_csv("data/train.csv")
    # test_data = pd.read_csv("data/test.csv")
    # print(test_data.describe())
    # data = pd.concat((train_data, test_data))
    # print(len(train_data), len(test_data), len(data))
    # print(data.describe())
    from functions import get_normalized_data
    train_data = get_normalized_data({"Date": "original", "Generation": "original"}, part="Train")
    print(train_data.describe())
    # print(train_data.values[-3:])
    return


def plot_scatter_1():
    from functions import get_normalized_data
    train_data = get_normalized_data({"Date": "original", "Module": "original"}, part="All")

    import matplotlib.pyplot as plt
    _, axs = plt.subplots(4, 1, figsize=(8, 9))

    import numpy as np
    import pandas as pd
    data = {}
    MODULES = ["MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3 320W", "AUO PM060MW3 325W"]
    for mi, module in enumerate(MODULES):
        data[module] = train_data[train_data["Module"]==module]
        data[module]["Date"] = pd.to_datetime(data[module]["Date"]) - pd.to_datetime(["2020/6/9"]*len(data[module]))
        x_time = np.array(data[module]["Date"], dtype=np.float64) / 86400000000000
        axs[mi].scatter(x_time, data[module]["Generation"], s=1, label="Generation", c="red")
        axs[mi].scatter(x_time, data[module]["Irradiance"], s=1, label="Irradiance", c="olive")
        # axs[mi].scatter(x_time, data[module]["Temp"], s=1, label="Temp", c="b")
        axs[mi].scatter(x_time, data[module]["Irradiance_m"], s=1, label="Irradiance_m", c="y")
        # axs[mi].scatter(x_time, data[module]["Temp_m"], s=1, label="Temp_m", c="c")
        axs[mi].legend()
        axs[mi].set_xlim([0, 617])
        axs[mi].set_title(module)
    plt.tight_layout()
    plt.show()
    return


def plot_scatter_2():
    from functions_new import get_normalized_data
    train_data = get_normalized_data(part="Train", module_setting="original")
    valid_data = get_normalized_data(part="Valid", module_setting="original")
    test_data  = get_normalized_data(part="Test", module_setting="original")

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import font_manager
    my_font = font_manager.FontProperties(fname="C:\Windows\Fonts\kaiu.ttf")
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(bottom=0.04, left=0.03, top=0.96, right=0.98)
    outers = gridspec.GridSpec(1, 3, wspace=0.15, hspace=0.3)

    MODULES = ["MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3"]
    for mi, module in enumerate(MODULES):
        ax = plt.Subplot(fig, outers[mi])
        ax.set_title(module, size=18)
        ax.axis("off")
        fig.add_subplot(ax)
        inners = gridspec.GridSpecFromSubplotSpec(7, 4, subplot_spec=outers[mi], wspace=0.35, hspace=0.25)
        for i, key in enumerate(["Irradiance", "Irradiance_m", "Temp", "Temp_m",
                                 # "Capacity", "Lat", "Lon", "Angle",
                                 "測站氣壓", "海平面氣壓", "測站最高氣壓", "測站最低氣壓",
                                 "氣溫", "最高氣溫", "最低氣溫", "露點溫度", "相對溼度", "最小相對溼度",
                                 "風速", "風向", "最大陣風", "最大陣風風向",
                                 "降水量", "降水時數", "日照時數", "日照率", "全天空日射量", "能見度",
                                 "日最高紫外線指數", "總雲量", "UV"]):
            ax = plt.Subplot(fig, inners[i])
            for data, color in [(train_data, "r"), (valid_data, "g"), (test_data, "b")]:
                data_module = data[data["Module"]==module]
                xs = data_module[key]
                ys = data_module["Generation"]
                xlabel = key
                ylabel = "Generation"
                ax.scatter(xs, ys, s=0.2, c=color)
            ax.set_xlabel(xlabel, size=10, fontproperties=my_font)
            ax.set_ylabel(ylabel, size=10, fontproperties=my_font)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(key, size=6)
            fig.add_subplot(ax)

    plt.show()
    return


def plot_3d_together():
    from functions_new import get_normalized_data
    data = get_normalized_data({"Date": "original", "Module": "original"}, part="All")
    data = data[data["Newer Module"].str.contains("AUO")]

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.animation as animation
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _  = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection="3d")

    irm = data["Irradiance_m"]
    irr = data["Irradiance"]
    gen = data["Generation"]
    ax.set_xlabel("Irradiance_m")
    ax.set_ylabel("Irradiance")

    tmp = data["Temp"]
    ax.scatter(irm, irr, gen, s=2, color=cm.cool(tmp))
    ax.set_zlabel("Generation")
    # ax.scatter(irm, irr, tmp, s=2, color=cm.plasma(gen))
    # ax.set_zlabel("Temp")

    # tpm = data["Temp_m"]
    # ax.scatter(irm, irr, gen, s=2, color=cm.cool(tpm))
    # ax.set_zlabel("Generation")
    # ax.scatter(irm, irr, tpm, s=2, color=cm.plasma(gen))
    # ax.set_zlabel("Temp_m")
    
    plt.tight_layout()

    # Save animation
    # def rotate(i):
    #     ax.view_init(30, i+45+90)
    #     return fig
    # anim = animation.FuncAnimation(fig, rotate, frames=360, interval=100, blit=False)
    # writer = animation.PillowWriter(fps=40)
    # anim.save("3D_scatter_together_i-im-t-gen.gif", writer=writer)

    plt.show()
    return


def plot_3d_split():
    from functions_new import get_normalized_data
    data = get_normalized_data(part="All", module_setting="original")

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.animation as animation
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(20, 10), subplot_kw=dict(projection="3d"))

    MODULES = ["MM60-6RT-300", "SEC-6M-60A-295", "AUO PM060MW3"]
    for mi, module in enumerate(MODULES):
        module_data = data[data["Module"]==module]
        irm = module_data["Irradiance_m"]
        irr = module_data["Irradiance"]
        gen = module_data["Generation"]
        tmp = module_data["Temp"]
        # tpm = module_data["Temp_m"]
        axs[mi].set_title(module, size=10)
        axs[mi].set_xlabel("Irradiance_m")
        axs[mi].set_ylabel("Irradiance")

        # axs[mi].scatter(irm, irr, tmp, color=cm.plasma(gen), s=2)
        # axs[mi].set_zlabel("Temp_m")
        axs[mi].scatter(irm, irr, gen, color=cm.cool(tmp), s=2)
        axs[mi].set_zlabel("Generation")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.05)
    plt.show()

    # Save
    # def rotate(i):
    #     for mi in range(len(MODULES)):
    #         axs[int(mi/2)][mi%2].view_init(30, i+45+90)
    #     return fig
    # anim = animation.FuncAnimation(fig, rotate, frames=360, interval=100, blit=False)
    # writer = animation.PillowWriter(fps=40)
    # anim.save("generation-temp_m.gif", writer=writer)
    return



if __name__ == "__main__":
    # describe()
    # plot_scatter_1()
    plot_scatter_2()
    # plot_3d_together()
    # plot_3d_split()