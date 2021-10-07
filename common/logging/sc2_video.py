import io
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.animation as animation
from tensorflow_probability import distributions as tfd


class AnimatedMnist:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, true, pred):
        # shape is [batch, step, unit, features]

        self.true = true
        self.pred = pred

        self.batch_size = true['unit_id'].shape[0]
        self.total_frames = true['unit_id'].shape[1]

        img_size = 6
        plots_per_sample = 2

        self.fig, self.ax = plt.subplots(self.batch_size, plots_per_sample, figsize=(plots_per_sample * img_size, self.batch_size * img_size))
        for ax in self.ax.flat:
            ax.axis([0, 1, 1, 0], )
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.subplots_adjust()

        self.stream = self.data_stream()

        self.artists = []

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000,
                                           blit=True, save_count=self.total_frames - 1, frames=self.total_frames - 1, repeat=False)

    def data_stream(self):
        for i in range(self.total_frames):
            id = self.true['unit_id'][:, i, :]
            id_h = self.pred['unit_id'][:, i, :]

            x = self.true['unit_continuous'][:, i, :, 5]
            y = self.true['unit_continuous'][:, i, :, 6]

            x_h = self.pred['unit_continuous'][:, i, :, 5]
            y_h = self.pred['unit_continuous'][:, i, :, 6]

            yield id, id_h, x, y, x_h, y_h

    def update(self, i):
        """Update the scatter plot."""
        id, id_h, x, y, x_h, y_h = next(self.stream)

        if len(self.artists) > 0:
            for art in self.artists:
                art.remove()

        for i in range(self.batch_size):
            self.artists = []

            unpadded_units = tf.cast(tf.not_equal(id[i, :, 0], 1), dtype=tf.int32)  # find indices where unit type not 0
            set_size = tf.reduce_sum(unpadded_units, axis=-1)

            self.artists.append(self.ax[i, 0].scatter(x[i, 0:set_size], y[i, 0:set_size], s=150, c='green'))
            ids = tf.argmax(id[i, 0:set_size], axis=-1)

            for j, txt in enumerate(ids):
                val = str(txt.numpy())
                self.artists.append(self.ax[i, 0].annotate(val, (x[i, j], y[i, j])))

            self.artists.append(self.ax[i, 1].scatter(x_h[i, 0:set_size], y_h[i, 0:set_size], s=150, c='green'))
            ids_h = tf.argmax(id_h[i, 0:set_size], axis=-1)
            for j, txt in enumerate(ids_h):
                val = str(txt.numpy())
                self.artists.append(self.ax[i, 1].annotate(val, (x_h[i, j], y_h[i, j])))

        return self.artists

    def save(self):
        self.ani.save('animation2.gif', writer='imagemagick', fps=1)


if __name__ == '__main__':
    id_dist = tfd.OneHotCategorical(tf.zeros([6, 20, 30, 190], dtype=tf.float32))
    continuous_dist = tfd.Independent(tfd.Normal(tf.zeros([6, 20, 30, 14], dtype=tf.float16), 0.5), 3)
    binary_dist = tfd.Independent(tfd.Bernoulli(tf.zeros([6, 20, 30, 48], dtype=tf.float16)), 3)

    true = {
        'unit_id': id_dist.sample(),
        'unit_continuous': continuous_dist.sample(),
        'unit_binary': binary_dist.sample()
    }

    pred = {
        'unit_id': id_dist.sample(),
        'unit_continuous': continuous_dist.sample(),
        'unit_binary': binary_dist.sample()
    }

    guy = AnimatedMnist(true, pred)
    guy.save()
