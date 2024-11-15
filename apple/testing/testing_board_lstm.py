import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import interp1d


@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class FastDrawingBoard:
    def __init__(self, size=(255, 255)):
        self.size = size
        self.board = np.ones(self.size) * 255  # white board

    def draw_stroke(self, start, end, color=0):
        """Draws a line using Bresenham's algorithm."""
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if 0 <= y1 < self.size[0] and 0 <= x1 < self.size[1]:
                self.board[y1, x1] = color
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

def rnn_model_prediction(points):
    model = tf.keras.models.load_model('../models/lstm_model.h5', custom_objects={'mse': mse})
    X = np.array([points])
    Y_pred = model.predict(X)
    return Y_pred

class FastInteractiveDrawingBoard(FastDrawingBoard):
    def __init__(self, size=(255, 255)):
        super().__init__(size)
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.board, cmap='gray', vmin=0, vmax=255)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.drawing = False
        self.last_pos = None
        self.points = []

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  
            self.drawing = True
            self.last_pos = (int(event.xdata), int(event.ydata))
            self.points.append(self.last_pos)

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  
            print('Drawing finished')
            
            self.drawing = False
            self.last_pos = None
            
            points_to_print = self.points.copy()
            self.points = []
            # make if for example there are 50 points or 10 points, or 1000 points we want to have only 6 points
            # let's use interpolation for that

            num_desired_points = 6
            if len(points_to_print) > num_desired_points:
                x = np.linspace(0, 1, len(points_to_print))
                f = interp1d(x, points_to_print, axis=0)
                x_new = np.linspace(0, 1, num_desired_points)
                points_to_print = f(x_new).tolist()
            print(points_to_print)
            for _ in range(35):
                ypred = rnn_model_prediction(points_to_print)
                points_to_print.append(ypred[0])
                for i in range(len(points_to_print) - 1):
                    plt.plot(
                        [points_to_print[i][0], points_to_print[i + 1][0]],
                        [points_to_print[i][1], points_to_print[i + 1][1]],
                        color='red'
                    )
                plt.xlim(0, 255)
                plt.ylim(0, 255)  
                plt.gca().invert_yaxis()  
                plt.draw()
                plt.pause(0.01)

            plt.xlim(0, 255)
            plt.ylim(0, 255)
            plt.gca().invert_yaxis()  
            plt.show()
        self.update_display()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        current_pos = (int(event.xdata), int(event.ydata))
        if self.last_pos is not None:
            self.draw_stroke(self.last_pos, current_pos)
        self.last_pos = current_pos
        self.points.append(current_pos)
        self.update_display()

    def update_display(self):
        """Update the display."""
        self.im.set_data(self.board)
        self.fig.canvas.draw_idle()

fast_interactive_board = FastInteractiveDrawingBoard()
plt.show()
