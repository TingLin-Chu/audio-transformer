# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import os
import math
import numpy as np
import resampy

import mel_features
import vggish_params

try:
    import soundfile as sf

    def wav_read(wav_file):
        wav_data, sr = sf.read(wav_file, dtype='int16')
        return wav_data, sr

except ImportError:

    def wav_read(wav_file):
        raise NotImplementedError(
            'WAV file reading requires soundfile package.')


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
      data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
      sample_rate: Sample rate of data.

    Returns:
      3-D np.array of shape [num_examples, num_frames, num_bands] which represents
      a sequence of examples, each of which contains a patch of log mel
      spectrogram, covering num_frames frames of audio and num_bands mel frequency
      bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # print(len(log_mel))

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))

    if len(log_mel) < example_window_length:
        while len(log_mel) < example_window_length:
            log_mel = np.vstack([log_mel, log_mel[-1]])

    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)
    return log_mel_examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

    Returns:
      See waveform_to_examples.
    """
    wav_data, sr = wav_read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples(samples, sr)


def wavfile_to_examples_clips(wav_file, num_clips):
    """Generate audio feature for each clip

    Args:
        wav_file (str): Path to .wav file
        num_clips (int): number of clips in a video

    Returns:
        List: A list of shape [num_clips, num_examples, num_frames, num_bands] containing audio features of every clip in a video, an element means an audio feature of a clip.
    """
    ret = []
    wav_data, sr = wav_read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    wav_len = len(wav_data)
    last_idx = 0
    for i in range(1, num_clips+1):
        cur_idx = math.floor(wav_len / num_clips * i) + 1
        cur_samples = samples[last_idx:cur_idx]
        # print(f"cur_samples len = {len(cur_samples)}")
        cur_mel = waveform_to_examples(cur_samples, sr)
        # print(f"cur_mel len = {len(cur_mel)}")
        ret.append(cur_mel)
        last_idx = cur_idx
    # print(f"ret len = {len(ret)}")
    return ret


if __name__ == "__main__":
    wav_folder = "F:\\XD_Violence\\Audios\\Training"
    npy_folder = "F:\\XD_Violence\\Mel_features\\Training"
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)
    num_clips = 32  # From RTFM
    for fn in os.listdir(wav_folder):
        wav_file = os.path.join(wav_folder, fn)
        npy_fn = fn.replace(".wav", ".npy")
        npy_path = os.path.join(npy_folder, npy_fn)
        if os.path.exists(npy_path):
            # print(f"Already extracted mel features from wav: {fn}")
            continue
        print(f"Converting wav: {fn}...")
        video_clip_features = wavfile_to_examples_clips(
            wav_file=wav_file, num_clips=num_clips)
        video_clip_features = np.array(video_clip_features)
        np.save(npy_path, np.array(video_clip_features))
    print(f"Successfully extracted mel features from wav in {wav_folder}!!!")
