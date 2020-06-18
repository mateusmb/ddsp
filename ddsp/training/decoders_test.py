# Copyright 2020 The DDSP Authors.
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

# Lint as: python3
"""Tests for ddsp.training.decoders."""

from absl.testing import parameterized
import ddsp.training.decoders as decoders
import ddsp.training.preprocessing as preprocessing
import numpy as np
import tensorflow.compat.v2 as tf


class DilatedConvStackDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    """Create some common default values for decoder."""
    super().setUp()
    # For decoder
    self.ch = 256
    self.layers_per_stack = 5
    self.stacks = 2
    self.output_splits = (('amps', 1), ('harmonic_distribution', 40),
                          ('noise_magnitudes', 65))

    # For audio features
    self.f0_hz_val = 440
    self.loudness_db_val = -50
    self.frame_rate = 250
    self.length_in_sec = 1
    self.time_steps = self.frame_rate * self.length_in_sec

  def _gen_dummy_audio_features(self):
    audio_features = {}
    audio_features['f0_hz'] = np.repeat(self.f0_hz_val,
                                        self.length_in_sec * self.frame_rate)
    audio_features['loudness_db'] = np.repeat(
        self.loudness_db_val, self.length_in_sec * self.frame_rate)
    return audio_features

  def test_correct_output_splits_and_shapes_conv_dilated_stack(self):
    preprocessor = preprocessing.DefaultPreprocessor(time_steps=self.time_steps)
    decoder = decoders.DilatedConvStackDecoder(
        ch=self.ch,
        layers_per_stack=self.layers_per_stack,
        stacks=self.stacks,
        output_splits=self.output_splits)

    audio_features = self._gen_dummy_audio_features()
    conditioning = preprocessor(audio_features)
    output = decoder(conditioning)
    for output_name, output_dim in self.output_splits:
      dummy_output = np.zeros((1, self.time_steps, output_dim))
      self.assertShapeEqual(dummy_output, output[output_name])


if __name__ == '__main__':
  tf.test.main()
