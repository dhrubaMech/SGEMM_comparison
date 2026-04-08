import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager


plt.rcParams['font.family'] = 'Britannic Bold'
# plt.rcParams['font.serif'] = ['Times New Roman']

labelFontsize = 18
tickFontsize = 15

def plotRectTiles():
    TS  = [16 , 32]
    TSM = [16, 32, 16, 16, 32, 32, 64]
    TSK = [16, 32, 32, 64, 64, 16, 32]

    timings = np.array([1726, 1335, 2227, 2180, 1373, 1225, 1115])

    f,ax = plt.subplots(1,1,figsize=(7,4))

    ax.barh(np.arange(len(TSM)),timings/1000,fc="silver")

    ax.set_yticks(np.arange(len(TSM)))
    ax.set_yticklabels(["16x16", "32x32", "16x32", "16x64", "32x64", "32x16", "64x32"])

    ax.set_ylabel("[TSM][TSK]",fontsize=labelFontsize)
    # ax.set_xlabel("t (microsecs)",fontsize=labelsize)
    ax.set_xlabel("t (ms)",fontsize=labelFontsize)

    ax.tick_params(labelsize=tickFontsize)

    plt.tight_layout()
    plt.savefig("EffectTilesize.png",dpi=300)
    plt.close()

def plot2Dregister():

    Nsample = 10
    TSM = np.array([16,32,64,128])
    TSK = np.array([16,32,64,128])
    WPT = [4,8]

    Ts = np.zeros((len(WPT),TSM.shape[0],TSK.shape[0],Nsample)) * np.nan

    f,ax = plt.subplots(1,2,figsize=(10,4))

    for w,wpt in enumerate(WPT):
        for m,tsm in enumerate(TSM):
            for k,tsk in enumerate(TSK):
                print(f"[{tsm}][{tsk}]")
                if (wpt == 4) and (tsm == 128) and (tsk in [16,32]):
                    continue
                try:
                    data = np.loadtxt(f"EffectTilesize/CUDA2Dregister_N11_TSM{tsm}_TSN{tsm}_TSK{tsk}_WPT{wpt}.csv",delimiter=",")
                    Ts[w,m,k,:] = data[1:]
                except:
                    print("  - no data")
                # print(data) ; exit()

    meanWPT4 = np.mean(Ts[0],axis=-1)
    meanWPT8 = np.mean(Ts[1],axis=-1)
    # print(meanWPT8.shape, meanWPT4.shape)

    ax[0].imshow(1/meanWPT4,cmap="cool",origin='lower')
    ax[1].imshow(1/meanWPT8,cmap="cool",origin='lower')

    for w in range(len(WPT)):
        ax[w].set_xlabel("TSK",fontsize=labelFontsize)
        ax[w].set_ylabel("TSM",fontsize=labelFontsize)

        ax[w].set_xticks(np.arange(TSK.shape[0]), labels=TSK,)
        ax[w].set_yticks(np.arange(TSM.shape[0]), labels=TSK,)

        ax[w].tick_params(labelsize=tickFontsize)
        
        ax[w].set_title(f"WPT = {WPT[w]}",fontsize=labelFontsize)
    
    for m in range(TSM.shape[0]):
        for k in range(TSK.shape[0]):
            ax[0].text(k,m, np.round(meanWPT4[m,k]/1000,2), ha="center", va="center", color="w", fontsize=labelFontsize*0.8)
            ax[1].text(k,m, np.round(meanWPT8[m,k]/1000,2), ha="center", va="center", color="w", fontsize=labelFontsize*0.8)


    plt.tight_layout()
    plt.savefig("EffectiveTileSize2Dregister.png",dpi=300)
    plt.close()


if __name__ == "__main__":

    # fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    # for font in fonts:
    #    print(font)  
    # exit()   

    # plotRectTiles()

    plot2Dregister()

    
