from __future__ import print_function

import os
import numpy as np

import numpy as np
import tensorflow.compat.v1 as tf

import vggish_params
import vggish_postprocess
import vggish_slim


flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', "vggish_model.ckpt",
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

# Path to input folder
mels_folder = "F:\\XD_Violence\\Mel_features\\Training_zero"
# Path to output folder
embeddings_folder = "F:\\XD_Violence\\Embeddings\\Training_zero"
if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)


def main():
    for f in os.listdir(mels_folder):
        # Check if the embedding already exists
        mels_path = os.path.join(mels_folder, f)
        embeddings_path = os.path.join(embeddings_folder, f)
        if os.path.exists(embeddings_path):
            print(f"Already extracted vggish embeddings: {f}")
            continue

        # Load mel feature of the video
        mel_features_video = np.load(mels_path)
        print(f"Converting to embedding: {f}......")
        embeddings_video = []
        for mel_feature_clip in mel_features_video:
            # Prepare a postprocessor to munge the model embeddings.
            pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

            with tf.Graph().as_default(), tf.Session() as sess:
                # Define the model in inference mode, load the checkpoint, and
                # locate input and output tensors.
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
                features_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.OUTPUT_TENSOR_NAME)

                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: mel_feature_clip})
                postprocessed_batch = pproc.postprocess(embedding_batch)

                # Append the clip embedding to the list
                embeddings_video.append(postprocessed_batch)

        np.save(embeddings_path, np.array(embeddings_video))


if __name__ == "__main__":
    main()
    print("Successfully converted features to embeddings!!!")
