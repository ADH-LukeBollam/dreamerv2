import tensorflow as tf
from tensorflow_probability import distributions as tfd
import cv2 as cv
import numpy as np


class Sc2Video:
    def __init__(self, img_res, video_batches):
        # shape is [batch, step, unit, features]

        self.img_res = img_res

        self.video_batches = video_batches

        self.batch_size = video_batches[0]['unit_id'].shape[0]
        self.total_frames = video_batches[0]['unit_id'].shape[1]
        self.num_videos = len(video_batches)

    def get_frames(self):
        frames = []

        for f in range(self.total_frames):
            frame = np.zeros((self.img_res * self.batch_size, self.img_res * self.num_videos, 1))

            for v in range(self.num_videos):
                x_offset = v * self.img_res

                for b in range(self.batch_size):
                    y_offset = f * self.img_res

                    id = np.array(tf.argmax(self.video_batches[v]['unit_id'][b, f, :, :], axis=-1))
                    x = np.array(self.video_batches[v]['unit_continuous'][b, f, :, 5])
                    y = np.array(self.video_batches[v]['unit_continuous'][b, f, :, 6])

                    unpadded_units = np.not_equal(id, 0).astype(int)  # find indices where unit type not 0
                    num_units = int(np.sum(unpadded_units, axis=-1))

                    for u in range(num_units):
                        unit_x = int(x[u] * self.img_res + x_offset)
                        unit_y = int(y[u] * self.img_res + y_offset)
                        cv.circle(frame, (unit_x, unit_y), 10, (255,), -1)
                        pass

            cv.imwrite('color_img.jpg', frame)
            cv.imshow("image", frame)
            cv.waitKey()

            frames.append(frame)


if __name__ == '__main__':
    id_dist = tfd.OneHotCategorical(tf.zeros([6, 20, 30, 190], dtype=tf.float32))
    continuous_dist = tfd.Independent(tfd.Normal(tf.zeros([6, 20, 30, 14], dtype=tf.float16), 0.5), 3)
    binary_dist = tfd.Independent(tfd.Bernoulli(tf.zeros([6, 20, 30, 48], dtype=tf.float16)), 3)

    true = {
        'unit_id': id_dist.sample(),
        'unit_continuous': continuous_dist.sample() + 0.5,
        'unit_binary': binary_dist.sample()
    }

    pred = {
        'unit_id': id_dist.sample(),
        'unit_continuous': continuous_dist.sample() + 0.5,
        'unit_binary': binary_dist.sample()
    }

    guy = Sc2Video(256, [true, pred])
    guy.get_frames()
