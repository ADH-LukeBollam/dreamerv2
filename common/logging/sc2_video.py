import io
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.animation as animation

class AnimatedMnist:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, pred, true):
        # shape is [batch, step, unit, features]
        self.pred = pred
        self.true = true

        self.batch_size = true.shape[0]
        self.total_frames = true.shape[1]

        img_size = 3
        plots_per_sample = 2

        self.fig, self.ax = plt.subplots(figsize=(plots_per_sample * img_size, self.batch_size * img_size))
        for i in range(self.batch_size * plots_per_sample):
            self.ax[i].axis([0, 1, 1, 0], )
            self.ax[i].set_xticks([])
            self.ax[i].set_yticks([])

        self.fig.subplots_adjust(left=0.3, right=0.7)


        self.stream = self.data_stream()

        self.scat = [None] * self.batch_size * plots_per_sample

        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=10,
                                           blit=True, save_count=self.total_frames)

    def data_stream(self):
        for i in range(self.total_frames):
            x = self.true['unit_continuous'][:, i, :, 5]
            y = self.true['unit_continuous'][:, i, :, 6]

            x_h = self.pred['unit_continuous'][:, i, :, 5]
            y_h = self.pred['unit_continuous'][:, i, :, 6]

            yield x, y, x_h, y_h

    def update(self, i):
        """Update the scatter plot."""
        x, y, x_h, y_h = next(self.stream)

        for i in range(self.batch_size):
            if self.scat is not None:
                self.scat.remove()

            self.scat[2 * i] = self.ax[2 * i].scatter(x, y, s=150, c='green')
            self.scat[2 * i + 1] = self.ax[2 * i + 1].scatter(x_h, y_h, s=150, c='green')

        return self.scat

    def save(self):
        self.ani.save('animation2.gif', writer='imagemagick', fps=30)



def UnitVideo(truth, model):
    x = truth['unit_continuous'][..., 5]
    y = truth['unit_continuous'][..., 6]

    x_h = model['unit_continuous'][..., 5]
    y_h = model['unit_continuous'][..., 6]


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def set_to_plot(true, predicted):

    img_size = 3
    plots_per_sample = 2
    figure = plt.figure(figsize=(img_size * plots_per_sample, num_elements * img_size))
    plt.grid(False)
    plt.tight_layout()

    x = truth['unit_continuous'][..., 5]
    y = truth['unit_continuous'][..., 6]

    x_h = model['unit_continuous'][..., 5]
    y_h = model['unit_continuous'][..., 6]

    # image
    for i in range(num_elements):

        # truth set
        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample)
        x = true[i, :, 1]
        y = true[i, :, 0]
        plt.scatter(x, y)
        plt.axis([0, 1, 1, 0])

        # predicted set
        plt.subplot(num_elements, plots_per_sample, i * plots_per_sample + 1)
        x = predicted[i, :, 1]
        y = predicted[i, :, 0]
        plt.scatter(x, y)
        plt.axis([0, 1, 1, 0])

    return figure