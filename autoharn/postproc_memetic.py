

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import animation, axes

import outconfig.settings as cfg

data = None
t = np.linspace(0, 200)

agent_cells = [
        [0, 4],     # Root row
        [1, 1], [1, 4], [1, 7],    # Row 1
        [2, 0], [2, 1], [2, 2],  [2, 3], [2, 4], [2, 5],  [2, 6], [2, 7],[2, 8]  # Row 3
    ]

filename = None

plt.figure()
fig, axs = plt.subplots(3,9,figsize=(50,20),sharex=True)
plt.gca().xaxis.get_major_locator().set_params(integer=True)

def animate_func(num):

    global fig
    global axs 

    # Setup Plot
    fig.suptitle("Agent Improvement Plot",fontsize=25)
    fig.tight_layout(pad=3.0)

    # Hide the plots that are not an agent
    for ax in axs.reshape(-1):
        ax.axis('off')

    # Print agent data
    for agent_no, cell in enumerate(agent_cells):

        # Print headings
        axs[cell[0], cell[1]].set_title( f"Agent {agent_no}", size=20)

        data_temp = data[ (data[:,0] == agent_no) & (data[:,1] < num*4+1) ] 
        axs[cell[0], cell[1]].plot( data_temp[:,1], data_temp[:,2], marker='D', c='black', label=f"Pocket Fitness")

        # Turn on (all turned off)
        axs[cell[0], cell[1]].axis('on')

    axs[0, 4].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    #tempfn = filename.replace("Master.log",f"Improve.{num}.png")
    #plt.savefig(tempfn)


def plot_agent_improve():

    results = pd.read_csv(f"{cfg.outSumDir}/results.csv")
    
    datasets = pd.unique(results['Dataset'])
    algorithms = pd.unique(results['Method'])
    algorithms = algorithms[[ "meme" in algorithms]]
    seeds = pd.unique(results['Seed'])
    splits = pd.unique(results['Split'])

    for dataset in datasets:
        for algorithm in algorithms:
            for split in splits:
                for seed  in seeds:

                    global filename
                    global data 

                    filename = f"{cfg.outResDir}/{algorithm}/{dataset}/{seed}/{split}/{seed}.Master.log"
                    events = pd.read_csv(filename)
                    data = []

                    # Print agent data
                    for agent_no, cell in enumerate(agent_cells):

                        agent_events = events[ (events.iloc[:,3] == agent_no) & (events.iloc[:,2] == "AgentSwap") ]
                        for _, evt in agent_events.iterrows():
                            
                            data.append([agent_no, evt[1], evt[5]])
                            data.append([agent_no, evt[1], evt[8]])
                            
                    data = np.array(data)

                    # Plotting the Animation

                    line_ani = animation.FuncAnimation(fig, animate_func, interval=4, frames=50)

                    writergif = animation.PillowWriter(fps=1)
                    line_ani.save(filename.replace("Master.log","Improve.gif"), writer=writergif)
                    
                    plt.close()    
                    
                    #plt.savefig(filename.replace("Master.log","Improve.png"))
                                

