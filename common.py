"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.

 modified by jsapas
"""

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import numpy as np
import re
import itertools

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
from tqdm.auto import tqdm
import cmsisdsp
from numpy import pi as PI

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "train.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="train.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.2"
########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("Anomaly detection baseline and other ARM based models\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def get_arm_spectrogram(waveform, window_size, step_size):

    hanning_window_f32 = np.zeros(window_size)
    for i in range(window_size):
        hanning_window_f32[i] = 0.5 * (1 - cmsisdsp.arm_cos_f32(2 * PI * i / window_size ))

    hanning_window_q15 = cmsisdsp.arm_float_to_q15(hanning_window_f32)

    rfftq15 = cmsisdsp.arm_rfft_instance_q15()
    status = cmsisdsp.arm_rfft_init_q15(rfftq15, window_size, 0, 1)

    num_frames = int(1 + (len(waveform) - window_size) // step_size)
    fft_size = int(window_size // 2 + 1)

    # Convert the audio to q15
    waveform_q15 = cmsisdsp.arm_float_to_q15(waveform)

    # Create empty spectrogram array
    spectrogram_q15 = np.empty((num_frames, fft_size), dtype = np.int16)

    start_index = 0

    for index in range(num_frames):
        # Take the window from the waveform.
        window = waveform_q15[start_index:start_index + window_size]

        # Apply the Hanning Window.
        window = cmsisdsp.arm_mult_q15(window, hanning_window_q15)

        # Calculate the FFT, shift by 7 according to docs
        window = cmsisdsp.arm_rfft_q15(rfftq15, window)

        # Take the absolute value of the FFT and add to the Spectrogram.
        spectrogram_q15[index] = cmsisdsp.arm_cmplx_mag_q15(window)[:fft_size]

        # Increase the start index of the window by the overlap amount.
        start_index += step_size

    # Convert to numpy output ready for keras
    return cmsisdsp.arm_q15_to_float(spectrogram_q15).reshape(num_frames,fft_size) * 512

def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)

    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def file_to_vector_array_spec(file_name,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    n = (n_fft//2 + 1)
    dims = n * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)

    D = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length))
    log_spectrogram = 20.0 / power * np.log10(D + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n * t: n * (t + 1)] = log_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def file_to_vector_array_spec_quant(file_name,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    quantized version of file_to_vector_array_spec()
    """
    # calculate the number of dimensions
    n = (n_fft//2 + 1)
    dims = n * frames

    y, sr = file_load(file_name)

    # generate spectrogram using cmsisdsp
    #D = np.abs(librosa.stft(y, n_fft=n_fft,  hop_length=hop_length))
    #log_spectrogram = 20.0 / power * np.log10(D + sys.float_info.epsilon)

    quant_spectrogram = get_arm_spectrogram(waveform=y, window_size=n_fft, step_size=hop_length).T

    # calculate total vector size
    vector_array_size = len(quant_spectrogram[0, :]) - frames + 1

    # skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n * t: n * (t + 1)] = quant_spectrogram[:, t: t + vector_array_size].T

    return vector_array


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data
    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
        
    dirs = [d for d in dirs if os.path.isdir(d)]

    if 'target' in param:
        def is_one_of_in(substrs, full_str):
            for s in substrs:
                if s in full_str: return True
            return False
        dirs = [d for d in dirs if is_one_of_in(param["target"], str(d))]

    return dirs

########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : np.array( np.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
            logger.info((f'Creating data for {len(file_list)} files: size={dataset.shape[0]}'
                         f', shape={dataset.shape[1:]}'))
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

def list_to_vector_array_spec(file_list,
                         msg="calc...",
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : np.array( np.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    n = (n_fft//2 + 1)
    dims = n * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array_spec(file_list[idx],
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
            logger.info((f'Creating data for {len(file_list)} files: size={dataset.shape[0]}'
                         f', shape={dataset.shape[1:]}'))
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

def list_to_vector_array_spec_quant(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    quantized version of list_to_vector_array_spec()
    """
    # calculate the number of dimensions
    n = (n_fft//2 + 1)
    dims = n * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array_spec_quant(file_list[idx],
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), np.int16)
            logger.info((f'Creating data for {len(file_list)} files: size={dataset.shape[0]}'
                         f', shape={dataset.shape[1:]}'))
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files
    return :
        train_files : list [ str ]
            file list for training
    """
    logger.info("target_dir : {}".format('/'.join(str(target_dir).split('/')[-2:])))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        logger.exception(f"{training_list_path} -> no_wav_file!!")

    logger.info("# of training samples : {num}".format(num=len(files)))
    return files

def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list

def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav",
                             mode=True):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################