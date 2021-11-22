import tensorflow as tf
from tensorflow_probability import distributions as tfd
import cv2 as cv
import numpy as np


class Sc2Video:
    @staticmethod
    def get_frames(img_res, video_data):
        batch_size = video_data[0]['unit_continuous'].shape[0]
        total_frames = video_data[0]['unit_continuous'].shape[1]
        num_video_columns = len(video_data)

        frames = []
        for f in range(total_frames):
            video_columns = []
            for v in range(num_video_columns):
                batch_screens = []
                for b in range(batch_size):
                    screen = np.zeros((img_res, img_res, 1))

                    id = np.array(tf.argmax(video_data[v]['unit_id'][b, f, :, :], axis=-1))
                    x = np.array(video_data[v]['unit_continuous'][b, f, :, 0]) + 0.5
                    y = np.array(video_data[v]['unit_continuous'][b, f, :, 1]) + 0.5

                    unpadded_units = np.not_equal(np.array(tf.argmax(video_data[0]['unit_id'][b, f, :, :], axis=-1)), 0).astype(int)  # find indices where unit type not 0
                    num_units = int(np.sum(unpadded_units, axis=-1))

                    for u in range(num_units):
                        unit_x = int(x[u] * img_res)
                        unit_y = int(y[u] * img_res)
                        cv.circle(screen, (unit_x, unit_y), 10, (0.4,), -1)
                        # unit_id = int(id[u])
                        # cv.putText(screen, str(unit_id), (unit_x, unit_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (1,), 2, cv.LINE_AA)

                    cv.rectangle(screen, (0, 0), (img_res, img_res), (0.3,), 2)

                    batch_screens.append(screen)
                column = np.concatenate(batch_screens, axis=0)
                video_columns.append(column)

            frame = np.concatenate(video_columns, axis=1)

            frames.append(frame)
        frames = np.stack(frames, axis=0)
        return frames


if __name__ == '__main__':
    example_batch_size = 3
    example_num_columns = 2
    example_num_units = 30

    x_points = tf.range(0, 1, 1.0/example_num_units, tf.float32)
    y_points = tf.range(0, 1, 1.0/example_num_units, tf.float32)
    img = []
    for i in range(20):
        x_points = tf.roll(x_points, 1, 0)
        points = tf.stack((x_points, y_points), 1)
        img.append(points)

    img = tf.stack(img, 0)
    xy = tf.pad(img, ((0, 0), (0, 0), (5, 7)), 'CONSTANT', 0)

    id_dist = tfd.OneHotCategorical(tf.zeros([1, 20, example_num_units, 190], dtype=tf.float32))
    binary_dist = tfd.Independent(tfd.Bernoulli(tf.zeros([1, 20, example_num_units, 48], dtype=tf.float16)), 3)

    true = {
        'unit_id': tf.tile(id_dist.sample(), (example_batch_size, 1, 1, 1)),
        'unit_continuous': tf.tile(tf.expand_dims(xy, 0), (example_batch_size, 1, 1, 1)),
        'unit_binary': tf.tile(binary_dist.sample(), (example_batch_size, 1, 1, 1))
    }

    frames = Sc2Video.get_frames(512, [true] * example_num_columns)

    for f in frames:
        cv.imwrite('color_img.jpg', f)
        cv.imshow("image", f)
        cv.waitKey()
