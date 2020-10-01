from pathlib import Path
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import python_for_imscroll.image_processing as imp
import python_for_imscroll.drift_correction as dcorr
from python_for_imscroll import mapping
from python_for_imscroll import utils

def test_auto_integration():
    test_data_dir = Path(__file__).parent.resolve() / 'test_data'
    blue_image_path = test_data_dir / '20200228/hwligroup00774'
    blue_image_sequence = imp.ImageSequence(blue_image_path)
    green_image_path = test_data_dir / '20200228/hwligroup00775'
    green_image_sequence = imp.ImageSequence(green_image_path)
    aoiinfo_path = test_data_dir / '20200228/L2_aoi.dat'
    aois = imp.Aois.from_imscroll_aoiinfo2(aoiinfo_path)
    parameter_file_path = test_data_dir / '20200228/20200228parameterFile.xlsx'
    channels_data = utils.read_excel(parameter_file_path, sheet_name='channels')
    channels_data = channels_data.loc[channels_data.order-1, ['name', 'map file name']]
    target_channel = channels_data.name[0]
    aois.channel = target_channel
    binder_channels_map = {row['name']: mapping.Mapper.from_imscroll(test_data_dir / '20200228' / (row['map file name'] + '.dat'))
                           for i, row in channels_data.iloc[1:, :].iterrows()}
    parameters = utils.read_excel(parameter_file_path, sheet_name='L2')

    intcorrected = sio.loadmat(test_data_dir / '20200228/L2_intcorrected.dat')
    true_intensity = intcorrected['aoifits']['beforeBackgroundblue'].item()
    true_background = intcorrected['aoifits']['backgroundTraceblue'].item()
    drifter = dcorr.DriftCorrector.from_imscroll(test_data_dir / '20200228/L2_driftlist.dat')
    intensity = np.zeros((blue_image_sequence.length, len(aois)))
    background = np.zeros((blue_image_sequence.length, len(aois)))
    for i, image in tqdm(enumerate(blue_image_sequence), total=blue_image_sequence.length):
        drifted_aois = drifter.shift_aois(aois, i)
        intensity[i, :] = drifted_aois.get_intensity(image)
        background[i, :] = drifted_aois.get_background_intensity(image)
    cond1 = np.isnan(intensity).any(axis=0)
    cond2 = np.isnan(background).any(axis=0)
    intensity = intensity[:, np.logical_not(np.logical_or(cond1, cond2))]
    background = background[:, np.logical_not(np.logical_or(cond1, cond2))]
    np.testing.assert_allclose(intensity, true_intensity.T)
    np.testing.assert_allclose(background, true_background.T, rtol=1e-3)

    intensity = np.zeros((blue_image_sequence.length, len(aois)))
    background = np.zeros((blue_image_sequence.length, len(aois)))
    for channel, mapper in binder_channels_map.items():
        for i, image in tqdm(enumerate(green_image_sequence), total=green_image_sequence.length):
            mapped_aois = mapper.map(aois, to_channel=channel)
            drifted_aois = drifter.shift_aois(mapped_aois, i)
            np.testing.assert_allclose(drifted_aois._coords,
                                       intcorrected['aoifits']['datagreen'].item()[i*len(aois):(i+1)*len(aois), [4, 3]]-1,
                                       rtol=1e-6)
            intensity[i, :] = drifted_aois.get_intensity(image)
            background[i, :] = drifted_aois.get_background_intensity(image)
    cond1 = np.isnan(intensity).any(axis=0)
    cond2 = np.isnan(background).any(axis=0)
    intensity = intensity[:, np.logical_not(np.logical_or(cond1, cond2))]
    background = background[:, np.logical_not(np.logical_or(cond1, cond2))]
    true_intensity = intcorrected['aoifits']['beforeBackgroundgreen'].item()
    true_background = intcorrected['aoifits']['backgroundTracegreen'].item()
    np.testing.assert_allclose(intensity, true_intensity.T, rtol=1e-6)
    np.testing.assert_allclose(background, true_background.T, rtol=7e-3)
