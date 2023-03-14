import math
import numpy as np
import cv2
from PIL import Image

try:
    from .globals_ import Player, Winner
except:
    pass


class BezierCurves:
    def __init__(self, num_points=100):
        self.num_points = num_points
        pass

    @staticmethod
    def binomial(i, n):
        """Binomial coefficient"""
        return math.factorial(n) / float(
            math.factorial(i) * math.factorial(n - i))

    def bernstein(self, t, i, n):
        """Bernstein polynom"""
        return self.binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))

    def bezier(self, t, points):
        """Calculate coordinate of a point in the bezier curve"""
        n = len(points) - 1
        x = y = 0
        for i, pos in enumerate(points):
            bern = self.bernstein(t, i, n)
            x += pos[0] * bern
            y += pos[1] * bern
        # return np.ceil(x), np.ceil(y)
        return int(x), int(y)

    def bezier_curve_range(self, n, points, return_type):
        """Range of points in a curve bezier"""
        coord_list = []
        for i in range(n):
            t = i / float(n - 1)
            coord_list.append(self.bezier(t, points))
        coord_list = np.array(coord_list)
        if return_type == 'xy':
            x = coord_list[:, 0]
            y = coord_list[:, 1]
            return x, y
        elif return_type == 'ndarray':
            return coord_list
        else:
            raise NotImplemented

    def __call__(self, points, return_type):
        return self.bezier_curve_range(self.num_points, points, return_type)


def init_target_image(num_samples):
    img = np.zeros([512, 512])
    x_coords = [0, 0, 100, 100, 0]
    y_coords = [0, 100, 100, 200, 200]
    # img = np.zeros([1200, 1200])
    # x_coords = [300, 700, 800, 200, 1100]
    # y_coords = [700, 500, 1000, 200, 700]

    im2draw = img.copy()
    color = 255
    thickness = 3

    points = np.stack([x_coords, y_coords]).T + 200
    num_points = max(int(max(points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min()) / 15), 2)

    drawer = BezierCurves(num_points=num_points)

    for i in range(num_samples):
        coordinates = drawer(points + 50 * i, return_type='ndarray')
        for idx in range(coordinates.shape[0] - 1):
            start_point = coordinates[idx]
            end_point = coordinates[idx + 1]
            im2draw = cv2.line(im2draw, start_point, end_point, color, thickness)
    return im2draw


class Environment(object):
    """The environment MuZero is interacting with."""

    def __init__(self, action_space=10, original_code=False):
        target_image = init_target_image(num_samples=12)
        self.done = False
        self.turn = 0
        self.winner = None
        self.resigned = False

        self.current_image = np.zeros_like(target_image)
        self.target_image = target_image
        self.original_code = original_code

        self.current_loss = self.calc_loss(self.current_image)
        self.previous_loss = 0
        self.current_reward = self.calc_reward(loss=self.current_loss, prev_loss=self.previous_loss)

        self.layers = {}
        self.action_space = action_space
        self.intersection_rate = 0

    def calc_loss(self, image, threshold=5):
        if self.original_code:
            loss = np.mean(np.power(self.target_image - image, 2))
        else:
            pred_mask = image > threshold
            gt_mask = self.target_image > threshold
            self.intersection_rate = max((pred_mask * gt_mask).sum(), 1) / gt_mask.size

            difference = np.abs(self.target_image - image)
            loss = np.mean(difference) / max(self.intersection_rate, .1)
            # @TODO pixel-wise loss calculation: ONLY for naive approach with Bezier curves
        return loss

    def calc_reward(self, loss, prev_loss):
        loss += 1e-5
        prev_loss += 1e-5
        if self.original_code:
            if loss < prev_loss:  # original code
                reward = (prev_loss - loss) / prev_loss
            else:
                reward = (prev_loss - loss) / loss
        else:
            reward = np.exp(np.abs(prev_loss - loss))

            if loss > prev_loss:  # set penalty - reverse reward
                reward = -reward
        return reward

    def draw_curve(self, points, thickness=3, color=1):
        # отрисовка кривой Безье
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        if len(points.shape) == 3:
            points = points[0]

        num_points = max(
            int(max(points[:, 0].max() - points[:, 0].min(), points[:, 1].max() - points[:, 1].min()) / 15), 2
        )
        drawer = BezierCurves(num_points)
        coordinates = drawer(points, return_type='ndarray')

        im2draw = self.current_image.copy()

        for idx in range(coordinates.shape[0] - 1):
            start_point = coordinates[idx]
            end_point = coordinates[idx + 1]
            im2draw = cv2.line(im2draw, start_point, end_point, color, thickness)
        return im2draw

    def reset(self):
        self.current_image = np.zeros_like(self.target_image)
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, current_image):
        self.current_image = current_image
        # self.turn = self.turn_n()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def legal_actions(self):
        return [i for i in range(self.action_space)]

    def step(self, points, thickness=3, color=255):
        prev_loss = self.calc_loss(self.current_image)

        # @TODO color loss
        color_loss = 0

        # @TODO thickness loss
        thickness_loss = 0

        # lines loss
        self.current_image = self.draw_curve(points.cpu().detach().numpy().astype(int), thickness, color)
        self.current_loss = color_loss + thickness_loss + self.calc_loss(self.current_image)
        if abs(self.current_loss) <= 0.1 or self.turn > 1e3:
            self.done = True

        reward = self.calc_reward(self.current_loss, self.previous_loss)
        self.previous_loss = self.current_loss
        self.turn += 1
        return reward

    def black_and_white_plane(self):
        layers = np.zeros([self.action_space, *self.current_image.shape])
        for layer in range(self.action_space):
            layers[layer] = np.zeros_like(self.current_image)

        # for i in range(6):
        #     for j in range(7):
        #         if self.board[i][j] == ' ':
        #             board_white[i][j] = 0
        #             board_black[i][j] = 0
        #         elif self.board[i][j] == 'X':
        #             board_white[i][j] = 1
        #             board_black[i][j] = 0
        #         else:
        #             board_white[i][j] = 0
        #             board_black[i][j] = 1
        #
        # return numpy.array(board_white), numpy.array(board_black)

    def player_turn(self):
        if self.turn % 2 == 0:
            return Player.white
        else:
            return Player.black

    def render(self, points, thickness=3, color=1):
        Image.fromarray(self.draw_curve(points, thickness, color)).show()


def main():
    from PIL import Image
    Image.fromarray(init_target_image(num_samples=1)).show()


if __name__ == '__main__':
    main()
