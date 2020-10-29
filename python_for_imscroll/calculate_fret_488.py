import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gui.image_view as iv

def main():
    # Set qt gui
    iv.set_qapplication()

    # Open intensity.npz file
    intensity_file_path = iv.open_file_path_dialog()
    intensity_file = np.load(intensity_file_path)
    channels = ['green-green', 'green-red', 'red-red']
    intensity = {channel: intensity_file[channel] for channel in channels}
    

    # Calculate fret
    fret = intensity['green-red']/(intensity['green-red'] + intensity['red-red'])


    # Save as csv
    np.savetxt(intensity_file_path.parent / 'fret.csv', fret, delimiter=',')
    for channel, channel_intensity in intensity.items():
        np.savetxt(str(intensity_file_path.with_suffix(''))+channel+'.csv',
                   channel_intensity,
                   delimiter=',')
    
    # Plot
    n_frames, n_tethers = fret.shape
    time = np.arange(n_frames)
    fig_dir = intensity_file_path.with_suffix('')
    if not fig_dir.is_dir():
        fig_dir.mkdir()

    for i_tether in tqdm(range(n_tethers)):
        fig, (ax_int, ax_tot_int, ax_fret, ax_pb) = plt.subplots(nrows=4)
        
        ax_int.plot(time, intensity['green-green'][:, i_tether], color='b')  # Donor
        ax_int.plot(time, intensity['green-red'][:, i_tether], color='g')  # Acceptor
        ax_int.set_ylim(bottom=-1000)

        ax_fret.plot(time, fret[:, i_tether], color='purple')
        ax_fret.set_ylim((0, 1))

        ax_tot_int.plot(time, (intensity['green-green'][:, i_tether]
                               + intensity['green-red'][:, i_tether]),
                        color='k')
        ax_tot_int.set_ylim(bottom=-1000)

        ax_pb.plot(time, intensity['red-red'][:, i_tether], color='orange')
        ax_pb.set_ylim(bottom=-1000)

        fig.savefig(fig_dir / '{}.svg'.format(i_tether))
        plt.close(fig)

    # Plot fret histogram
    fig_path = intensity_file_path.with_name(intensity_file_path.stem + '_fret_histogram.svg')
    fig, ax = plt.subplots()
    ax.hist(fret.mean(axis=0), bins=np.linspace(start=0, stop=1, num=20))
    ax.set_xlim((0, 1))
    fig.savefig(fig_path)
    plt.close(fig)
    

    



if __name__ == '__main__':
    main()