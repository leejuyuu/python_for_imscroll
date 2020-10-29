
from collections import namedtuple
from pathlib import Path
import numpy as np
import python_for_imscroll.image_processing as imp
import python_for_imscroll.mapping as mapping
import gui.image_view as iv

Channel = namedtuple('Channel', ['ex', 'em'])

def main():
    iv.set_qapplication()
    image_dir = iv.select_directory_dialog()
    sequence_order = [Channel(ex='green', em='green'),
                      Channel(ex='green', em='red'),
                      Channel(ex='red', em='red')]
    image_sub_dirs = sorted((path for path in image_dir.iterdir() if path.is_dir()))
    if len(image_sub_dirs) == len(sequence_order):
        sequences = {channel: imp.ImageSequence(image_sub_dir)
                     for channel, image_sub_dir in zip(sequence_order, image_sub_dirs)}
    else:
        raise ValueError('Wrong number of glimpse directories.')
    mapping_file_path = iv.open_file_path_dialog()
    mapper = mapping.Mapper.from_imscroll(mapping_file_path)
  
    aois_path = iv.open_file_path_dialog()
    aois = imp.Aois.from_imscroll_aoiinfo2(aois_path)
    aois.channel = 'red'

    traces = dict()
    for channel, image_sequence in sequences.items():
        if channel.em != aois.channel:
            channel_aois = mapper.map(aois, to_channel=channel.em)
        else:
            channel_aois = aois
        
        intensity = np.zeros((image_sequence.length, len(aois)))
        for frame, image in enumerate(image_sequence):
            raw_intensity = channel_aois.get_intensity(image)
            background = channel_aois.get_background_intensity(image)
            intensity[frame, :] = raw_intensity - background
        traces[channel] = intensity
    traces_str_key = {f'{channel.ex}-{channel.em}': intensity
                     for channel, intensity in traces.items()}

    save_path = iv.save_file_path_dialog()
    np.savez(save_path, **traces_str_key)



if __name__ == '__main__':
    main()
