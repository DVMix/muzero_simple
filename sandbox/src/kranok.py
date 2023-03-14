from abc import ABCMeta, abstractmethod, abstractproperty
from math import *
import copy
import random

import numpy

__author__ = 'off999555'

LINE_LENGTH = 10  # determine the approximated length for each portion of the curve, the less = curvy curve
T_PRECISION = 0.01  # the acceptable t difference when generating points on a curve
TOTAL_POINTS = 9  # total points to generate when calculating curve length (excluding the start and end point)


class ControlPointDetail(object):
    def __init__(self, wobbling_dist=None, perpendicular_dist=None, angle=None, angular_dist=None):
        assert (any(arg is not None for arg in (wobbling_dist, perpendicular_dist, angle, angular_dist)))
        if wobbling_dist is not None:
            self._wobbling_dist = wobbling_dist
            self._perpendicular_dist = perpendicular_dist
            self.update_angle()
        elif angle is not None:
            self._angle = angle
            self._angular_dist = angular_dist
            self.update_wobbling_dist()

    @property
    def wobbling_dist(self):
        return self._wobbling_dist

    @wobbling_dist.setter
    def wobbling_dist(self, value):
        self._wobbling_dist = value
        self.update_angle()

    @property
    def perpendicular_dist(self):
        return self._perpendicular_dist

    @perpendicular_dist.setter
    def perpendicular_dist(self, value):
        self._perpendicular_dist = value
        self.update_angle()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = angle_trunc(value)
        self.update_wobbling_dist()

    @property
    def angular_dist(self):
        return self._angular_dist

    @angular_dist.setter
    def angular_dist(self, value):
        self._angular_dist = value
        self.update_wobbling_dist()

    def update_angle(self):
        origin = (0, 0)
        destination = (self.wobbling_dist, self.perpendicular_dist)
        self._angle = get_angle_between_points(origin, destination)
        self._angular_dist = get_distance(origin, destination)

    def update_wobbling_dist(self):
        origin = (0, 0)
        destination = get_point_away_from(origin, self.angle, self.angular_dist)
        self._wobbling_dist = destination[0]
        self._perpendicular_dist = destination[1]

    def get_coordinate(self, start, end):
        """
        :type start: (float, float)
        :type end: (float, float)
        :rtype: (float, float)
        """
        midpoint = get_middle_point([start, end])
        dist = get_distance(start, end)
        angle = get_angle_between_points(start, end)
        return get_point_away_from(midpoint, angle + self.angle, dist * self.angular_dist)

    def flip(self):
        self.perpendicular_dist *= -1

    def reverse(self):
        self.wobbling_dist *= -1


class ICurve(object):
    """
    An abstract curve interface that every discrete curve classes must follow its implementations.
    Currently, the subclasses of ICurve are Curve and CurveCollection.

    Classes that need to implement this interface should be about continuous curve instantiation.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def start_point(self):
        """
        :rtype: (float, float)
        """

    @start_point.setter
    def start_point(self, value):
        pass

    @abstractproperty
    def end_point(self):
        """
        :rtype: (float, float)
        """

    @end_point.setter
    def end_point(self, value):
        pass

    @abstractproperty
    def total_points(self):
        """
        Get approximated number of total points to generate the curve when drawing or comparing to other curves

        :rtype: int
        """

    @abstractmethod
    def get_point_by_percentile(self, percentile):
        """
        :type percentile: float
        :rtype: (float, float)
        """

    @abstractmethod
    def get_points(self, total_points=None):
        """
        :param total_points: total points for each individual curve part
        :type total_points: float
        :rtype: list[(float, float)]
        """

    # TODO implement these methods
    @abstractmethod
    def flip(self):
        pass

    @abstractmethod
    def reverse(self):
        pass

    @abstractmethod
    def rotate(self, pivot_point, rotation_angle):
        """
        :type pivot_point: (float, float)
        :type rotation_angle: float
        """

    @abstractmethod
    def set_scale(self, scale, reference_point=None):
        """Scale the size of the curve according to your needs by adjusting the start point and end point of the curve
        :param scale: scale factor, default is 1, put negative value to flip the curve around a reference_point
        :type scale: float
        :param reference_point: reference_point to scale away or scale into (default is middle point of the curve)
        :type reference_point: (float, float)
        """

    @abstractmethod
    def translate(self, angle, distance):
        """
        :type angle: float
        :type distance: float
        """

    @abstractmethod
    def get_length(self, total_points=None):
        """
        :param total_points: increase this value so that you will get more accurate curve length
        :type total_points: int
        :return: curve length
        :rtype: float
        """

    @abstractmethod
    def get_angle(self):
        """
        :rtype: float
        """

    @abstractmethod
    def get_angle_by_percentile(self, percentile):
        """
        :type percentile: float
        :rtype: float
        """

    @abstractmethod
    def get_traced_points(self, orig_structure, target_structure=None, cts=None):
        """
        :param orig_structure: Original tracing structure
        :type orig_structure: ICurve
        :param target_structure: Target tracing structure, defaults to original tracing structure if not provided
        :type target_structure: ICurve
        :type cts: CurveTracingSettings
        """

    @abstractmethod
    def get_closest_point(self, point):
        """
        :type point: (float, float)
        :rtype: (float, float)
        """

    @abstractmethod
    def get_middle_point(self, total_points=None):
        """
        :type total_points: int
        :rtype: (float, float)
        """


class Curve(ICurve):
    """
    Store information of one curve which includes its control point details.

    :type control_point_details: list[ControlPointDetail]
    """

    def __init__(self, start, end):
        """
        :param start: start point
        :type start: (float, float)
        :param end: end point
        :type end: (float, float)
        """
        self.start_point = start
        self.end_point = end
        self.control_point_details = []

    @staticmethod
    def extract_curve(cpoints, cp_no=2, total_points=None, total_dir=8):
        """
        Get extracted points that are start, control and end points from all example points in a curve
        by moving points and minimizing error. This is a revised version, runs super fast!

        :param cpoints: sample points in an unknown curve
        :type cpoints: list
        :param cp_no: number of control points you want for your bezier curve that is going to be generated
        :type cp_no: int
        :param total_points: number of points (excluding start and end) to be generated when comparing error,
        :type total_points: int
            (put None for automation)
        :param total_dir: total number of directions to move each control point (must be greater than 2)
        :type total_dir: int
        :return: a curve along with extracted control point details
        :rtype: Curve
        """
        assert cp_no >= 0 and len(cpoints) >= 2 and dir > 2
        curve = Curve(cpoints[0], cpoints[-1])
        if len(cpoints) == 2 or cp_no == 0:
            return curve
        for i in range(cp_no):
            index_percentile = float(i + 1) / (cp_no + 1)
            index = int(index_percentile * len(cpoints))
            curve.add_control_point(cpoints[index])
        cpoints2 = curve.get_points(total_points)
        per1 = get_percentile_list(cpoints)
        per2 = get_percentile_list(cpoints2)
        base_error = get_perpendicular_curve_error(cpoints, cpoints2, per1, per2)
        # start moving rate for the control point
        average_perpendicular_dist = abs(sum(cpd.perpendicular_dist for cpd in curve.control_point_details)) / cp_no
        move_rate = 2 * average_perpendicular_dist
        move_rate_factor = 0.9  # multiplier when need more precision, must be less than 1
        minimum_move_rate = 0.01  # the least move rate possible to move the control point
        # print 'initial moving rate:', move_rate
        # print 'initial error:', base_error
        while move_rate > minimum_move_rate:
            # print 'move rate:', move_rate
            for cpd in curve.control_point_details:
                need_another_move = True
                while need_another_move:
                    need_another_move = False
                    wobbling_dist_tmp = cpd.wobbling_dist
                    perpendicular_dist_tmp = cpd.perpendicular_dist
                    for i in range(total_dir):
                        # x and y are just direction factor ordering from right, down, left, and up respectively
                        x, y = get_point_away_from((0, 0), 2 * pi * i / total_dir, 1)
                        cpd.wobbling_dist = wobbling_dist_tmp + move_rate * x
                        cpd.perpendicular_dist = perpendicular_dist_tmp + move_rate * y
                        cpoints2 = curve.get_points(total_points)
                        per2 = get_percentile_list(cpoints2)
                        new_error = get_perpendicular_curve_error(cpoints, cpoints2, per1, per2)
                        if new_error < base_error:
                            base_error = new_error
                            # print 'dir:',i,'error:',new_error
                            need_another_move = True
                            break
                    if not need_another_move:
                        cpd.wobbling_dist = wobbling_dist_tmp
                        cpd.perpendicular_dist = perpendicular_dist_tmp
            move_rate *= move_rate_factor
        # print 'reduced error:', base_error
        return curve

    def rotate(self, pivot_point, rotation_angle):
        """
        :type pivot_point: (float, float)
        :type rotation_angle: float
        """
        dist = get_distance(pivot_point, self.start_point)
        angle = get_angle_between_points(pivot_point, self.start_point)
        self.start_point = get_point_away_from(pivot_point, angle + rotation_angle, dist)

        dist = get_distance(pivot_point, self.end_point)
        angle = get_angle_between_points(pivot_point, self.end_point)
        self.end_point = get_point_away_from(pivot_point, angle + rotation_angle, dist)

    @property
    def start_point(self):
        return self._start_point

    @start_point.setter
    def start_point(self, value):
        self._start_point = value

    @property
    def end_point(self):
        """
        :rtype: (float, float)
        """
        return self._end_point

    @end_point.setter
    def end_point(self, value):
        self._end_point = value

    @property
    def main_points(self):
        """
        :rtype: list[(float, float)]
        """
        mpoints = [self.start_point]
        mpoints.extend(self.control_points)
        mpoints.append(self.end_point)
        return mpoints

    @property
    def control_points(self):
        """
        :rtype: list[(float, float)]
        """
        return [self.get_control_point(i) for i in range(len(self.control_point_details))]

    @property
    def total_points(self):
        """Get approximated number of total points to generate the curve when drawing
        :rtype: int
        """
        return get_total_points(self.main_points)

    def get_control_point(self, index):
        return self.control_point_details[index].get_coordinate(self.start_point, self.end_point)

    def get_point(self, t):
        return get_bezier_point(self.main_points, t)

    def get_point_by_percentile(self, percentile):
        return self.get_point(self.get_t(percentile))

    def get_points(self, total_points=None, include_start_end=True):
        return get_bezier_points(self.main_points, total_points, include_start_end)

    def flip(self):
        for control_point in self.control_point_details:
            control_point.flip()

    def reverse(self):
        for control_point_detail in self.control_point_details:
            control_point_detail.reverse()
        self.control_point_details.reverse()

    def translate(self, angle, distance):
        self.start_point = get_point_away_from(self.start_point, angle, distance)
        self.end_point = get_point_away_from(self.end_point, angle, distance)

    def add_control_point_detail(self, cpd):
        """
        :type cpd: ControlPointDetail
        """
        self.control_point_details.append(copy.deepcopy(cpd))

    def add_control_point(self, cp):
        self.control_point_details.append(self.extract_control_point_detail(cp))

    def get_length(self, total_points=None):
        """
        :param total_points: increase this value so that you will get more accurate curve length
        :type total_points: int
        :return: curve length
        :rtype: float
        """
        return get_curve_length(self.get_points(total_points))

    def get_angle(self, t=None):
        if t is None or not self.control_point_details:
            return get_angle_between_points(self.start_point, self.end_point)
        # current_point = self.get_point(t)
        previous_point = self.get_point(t - T_PRECISION / 2)
        next_point = self.get_point(t + T_PRECISION / 2)
        return get_angle_between_points(previous_point, next_point)

    def get_angle_by_percentile(self, percentile):
        t = self.get_t(percentile)
        if t < T_PRECISION / 2:
            t = T_PRECISION / 2
        elif t > 1 - T_PRECISION / 2:
            t = 1 - T_PRECISION / 2
        return self.get_angle(t)

    def extract_control_point_detail(self, control_point):
        """
        :rtype: ControlPointDetail
        """
        midpoint = get_middle_point([self.start_point, self.end_point])
        angle = get_angle_between_points(midpoint, control_point) - self.get_angle()
        angular_dist = get_distance(midpoint, control_point) / get_distance(self.start_point, self.end_point)
        return ControlPointDetail(angle=angle, angular_dist=angular_dist)

    def get_t(self, percentile):
        percentile = check_percentile(percentile)
        if percentile == 0 or percentile == 1:
            return percentile
        total_points = 1 / T_PRECISION
        points = self.get_points(total_points)
        accumulative_length = 0.0
        percentile_list = [0.0]
        for i in range(1, len(points)):
            accumulative_length += get_distance(points[i], points[i - 1])
            percentile_list.append(accumulative_length)
        for i in range(len(percentile_list)):
            percentile_list[i] /= accumulative_length
            if percentile <= percentile_list[i]:
                previous_t = (i - 1) / (total_points + 1)
                current_t = i / (total_points + 1)
                tilted_ratio = (percentile - percentile_list[i - 1]) / (percentile_list[i] - percentile_list[i - 1])
                t = previous_t + (current_t - previous_t) * tilted_ratio
                # print percentile, t
                return t

    def set_scale(self, scale, reference_point=None):
        """Scale the size of the curve according to your needs by adjusting the start point and end point of the curve
        :param scale: scale factor, default is 1, put negative value to flip the curve around a reference_point
        :type scale: float
        :param reference_point: reference_point to scale away or scale into (default is middle point of the curve)
        :type reference_point: (float, float)
        """
        if reference_point is None:
            reference_point = self.get_middle_point()
        start_dist = get_distance(self.start_point, reference_point)
        end_dist = get_distance(self.end_point, reference_point)
        start_angle = get_angle_between_points(reference_point, self.start_point)
        end_angle = get_angle_between_points(reference_point, self.end_point)
        self.start_point = get_point_away_from(reference_point, start_angle, start_dist * scale)
        self.end_point = get_point_away_from(reference_point, end_angle, end_dist * scale)

    def get_closest_point(self, reference_point):
        return get_closest_point(self.get_points(), reference_point)

    def get_middle_point(self, total_points=None):
        """
        :type total_points: int
        :rtype: (float, float)
        """
        return get_middle_point(self.get_points(total_points))

    def get_traced_points(self, orig_structure, target_structure=None, cts=None):
        """
        :param orig_structure: Original tracing structure
        :type orig_structure: ICurve
        :param target_structure: Target tracing structure, defaults to original tracing structure if not provided
        :type target_structure: ICurve
        :type cts: CurveTracingSettings
        """
        if target_structure is None:
            target_structure = orig_structure
        if target_structure is orig_structure:
            length_ratio = 1
        else:
            length_ratio = target_structure.get_length() / orig_structure.get_length()
        if cts is None:
            cts = CurveTracingSettings()
        cpoints = []
        integrity = 0.4  # this expression make the width multiplier only 40% tense in importance to alter total_points
        width_multiplier = integrity * (abs(cts.width_scale) - 1) + 1
        # print 'Width multiplier =', width_multiplier
        target_total_points = int(self.total_points * length_ratio * width_multiplier) + 1
        for i in range(target_total_points + 1):
            p = float(i) / target_total_points
            tracing_p = cts.start_percentile + p * (cts.end_percentile - cts.start_percentile)
            perpendicular_point = orig_structure.get_point_by_percentile(tracing_p)
            curve_point = self.get_point_by_percentile(p)
            # transforming into a new tracing structure
            new_perpendicular_point = target_structure.get_point_by_percentile(tracing_p)
            new_perpendicular_angle = cts.rotation_angle
            if not cts.fixed_angle:
                new_derived_angle = target_structure.get_angle_by_percentile(tracing_p)
                derived_angle = orig_structure.get_angle_by_percentile(tracing_p)
                perpendicular_angle = get_angle_between_points(perpendicular_point, curve_point)
                theta = get_difference_between_angles(derived_angle, perpendicular_angle)
                new_perpendicular_angle += new_derived_angle - theta
            distance = get_distance(perpendicular_point, curve_point)
            new_point = get_point_away_from(new_perpendicular_point, new_perpendicular_angle, distance * length_ratio)
            # altering perpendicular distance of new_point
            real_perpendicular_point = get_perpendicular_point(target_structure.start_point,
                                                               target_structure.end_point, new_point)
            dist = get_distance(real_perpendicular_point, new_point)
            if cts.fixed_width and dist:
                old_dist = get_distance(get_perpendicular_point(orig_structure.start_point, orig_structure.end_point,
                                                                curve_point), curve_point)
                old_ratio = old_dist / dist
            else:
                old_ratio = 1
            if not is_approx_equal(cts.width_scale * old_ratio, 1):
                angle = get_angle_between_points(real_perpendicular_point, new_point)
                new_dist = dist * cts.width_scale * old_ratio
                new_point = get_point_away_from(real_perpendicular_point, angle, new_dist)
            # import drawing
            # drawing.draw_line(perpendicular_point, curve_point, False)
            # drawing.draw_line(new_perpendicular_point, new_point, False)
            cpoints.append(new_point)
        return cpoints


class CurveCollection(ICurve):
    """
    Store connected curve parts. Support recursive storage, so a part can be the CurveCollection itself.

    :type parts: list[ICurve]
    """

    def __init__(self, *args):
        """
        :type args: list[ICurve]
        """
        self.parts = []
        for arg in args:
            assert isinstance(arg, ICurve)
            self.parts.append(arg)

    @property
    def start_point(self):
        return self.parts[0].start_point

    @start_point.setter
    def start_point(self, value):
        # todo implement the behavior of start_point setter
        raise NotImplementedError

    @property
    def end_point(self):
        return self.parts[-1].end_point

    @end_point.setter
    def end_point(self, value):
        # todo implement the behavior of end_point setter
        raise NotImplementedError

    @property
    def percentile_list(self):
        """
        :rtype: list[float]
        """
        percentile_list = [0]  # 0 is used for first slot so that we can check if percentile is between 0..1
        total_length = 0
        for part in self.parts:
            length = part.get_length()
            total_length += length
            percentile_list.append(total_length)
        for i in range(1, len(percentile_list)):
            percentile_list[i] /= total_length
        return percentile_list

    @property
    def total_points(self):
        """
        Get approximated number of total points to generate the curve when drawing

        :rtype: int
        """
        return sum(curve.total_points for curve in self.parts)

    def get_points(self, total_points=None):
        """
        :param total_points: total points for each individual curve part
        :type total_points: int
        :rtype: list[(float, float)]
        """
        total_points_list = self.get_total_points_list(total_points)
        return [point for i, curve in enumerate(self.parts) for point in
                curve.get_points(total_points_list[i])[:-1]] + [self.end_point]

    def get_total_points_list(self, total_points=None):
        """
        Distribute total_points properly into a list proportionally to the length of each part e.g. 10 => [2,3,5]

        :type total_points: int
        :rtype: list[int]
        """
        total_points_list = [curve.total_points for curve in self.parts]
        if total_points is not None:
            orig_total_points = sum(total_points_list)
            ratio = float(total_points) / orig_total_points
            total_points_list = [int(round(element * ratio)) for element in total_points_list]
        return total_points_list

    def get_point_by_percentile(self, percentile):
        """
        :type percentile: float
        :rtype: (float, float)
        """
        percentile = check_percentile(percentile)
        part, new_percentile = self.get_part_and_percentile(percentile)
        return part.get_point_by_percentile(new_percentile)

    def get_part_and_percentile(self, percentile):
        """
        :type percentile: float
        :rtype: (ICurve, float)
        """
        percentile = check_percentile(percentile)
        percentile_list = self.percentile_list
        for i in range(len(self.parts)):
            current_percentile = percentile_list[i]
            next_percentile = percentile_list[i + 1]
            if current_percentile <= percentile <= next_percentile:
                # new percentile gathered by extrapolation, measuring ratio between two lengths
                new_percentile = (percentile - current_percentile) / (next_percentile - current_percentile)
                return self.parts[i], new_percentile

    def get_length(self, total_points=None):
        """
        :type total_points: int
        :rtype: float
        """
        total_points_list = self.get_total_points_list(total_points)
        return sum(curve.get_length(total_points_list[i]) for i, curve in enumerate(self.parts))

    def get_closest_point(self, point):
        """
        :type point: (float, float)
        :rtype: (float, float)
        """
        closest_points = [curve.get_closest_point(point) for curve in self.parts]
        return get_closest_point(closest_points, point)

    def get_angle(self):
        """
        :rtype: float
        """
        return get_angle_between_points(self.start_point, self.end_point)

    def get_angle_by_percentile(self, percentile):
        """
        :type percentile: float
        :rtype: float
        """
        part, new_percentile = self.get_part_and_percentile(percentile)
        return part.get_angle_by_percentile(new_percentile)

    def set_scale(self, scale, reference_point=None):
        """
        :type scale: float
        :type reference_point: (float, float)
        """
        if reference_point is None:
            reference_point = self.get_middle_point()
        for curve in self.parts:
            curve.set_scale(scale, reference_point)

    def get_middle_point(self, total_points=None):
        """
        :type total_points: int
        :rtype: (float, float)
        """
        return get_middle_point(self.get_points(total_points))

    @staticmethod
    def extract_curve(cpoints, total_curves=None, cp_no=2, total_points=None, total_dir=8, allow_straight_line=False):
        """
        Extract several curves from a point list

        :param cpoints: a list of points on a 2D plane
        :type cpoints: list[(float, float)]
        :param total_curves: total number of curves, put nothing to let our algorithm estimate this automatically
        :type total_curves: int
        :param cp_no: number of control points for each curve
        :type cp_no: int
        :param total_points: desired number of points to generate for each curve
        :type total_points: int
        :param total_dir: total number of directions to move the control points when fitting the curves
        :type total_dir: int
        :param allow_straight_line: True if you want the curves to be as similar as possible to the sample points,
            False if you want more curvy curves with less accurate curve fitting precision
        :type allow_straight_line: bool
        :return: a CurveCollection collecting extracted curves
        :rtype: CurveCollection
        """
        if len(cpoints) == 2:
            print("Since there is only 2 points, control point extraction isn't needed.")
            return CurveCollection(Curve(cpoints[0], cpoints[1]))
        angles = get_angle_list(cpoints)
        derivatives = get_angle_derivative_list(angles)  # changes of angles (2nd derivative of points)
        if total_curves is None or total_curves < 1:
            print('curve length:', get_curve_length(cpoints))
            total_curves = get_total_peaks(derivatives[1:-1]) + 1
            print('total curves =', total_curves)
        # print 'Showing angles and their derivatives in degrees'
        # for i, thing in enumerate(
        #         [(degrees(angle), degrees(derivative)) for angle, derivative in zip(angles, derivatives)]):
        #     print i, thing
        get_abs_der = lambda t: abs(t[1])
        sorted_derivatives = sorted(enumerate(derivatives[1:-1], 1), key=get_abs_der)
        critical_indexes = []
        # gathering critical_indexes
        for i in range(total_curves - 1):
            while sorted_derivatives:
                high_der_index, high_der = sorted_derivatives.pop()
                if allow_straight_line or not any(abs(ci - high_der_index) <= 1 for ci in critical_indexes):
                    # print 'critical index:', high_der_index, degrees(high_der)
                    critical_indexes.append(high_der_index)
                    break
            if not sorted_derivatives:
                break
        critical_indexes.sort()
        critical_indexes.append(len(cpoints) - 1)
        # separate curve by critical_indexes
        cc = CurveCollection()
        previous_index = 0
        for end_index in critical_indexes:
            start_index = previous_index
            sliced_cpoints = cpoints[start_index:end_index + 1]
            curve = Curve.extract_curve(sliced_cpoints, cp_no, total_points, total_dir)
            cc.parts.append(curve)
            previous_index = end_index
        return cc

    def get_traced_points(self, orig_structure, target_structure=None, cts=None):
        """
        :param orig_structure: Original tracing structure
        :type orig_structure: ICurve
        :param target_structure: Target tracing structure, defaults to original tracing structure if not provided
        :type target_structure: ICurve
        :type cts: CurveTracingSettings
        """
        if cts is None:
            cts = CurveTracingSettings()
        cpoints = []
        # make the percentile collapse into desired length
        # we will generate all points comparing to a section of the tracing structure
        percentile_list = [cts.start_percentile + p * (cts.end_percentile - cts.start_percentile) for p in
                           self.percentile_list]
        for i in range(len(self.parts)):
            start_p = percentile_list[i]
            end_p = percentile_list[i + 1]
            cts = copy.copy(cts)
            cts.start_percentile = start_p
            cts.end_percentile = end_p
            cpoints.extend(self.parts[i].get_traced_points(orig_structure, target_structure, cts))
        return cpoints

    def flip(self):
        for part in self.parts:
            part.flip()

    def translate(self, angle, distance):
        for part in self.parts:
            part.translate(angle, distance)

    def reverse(self):
        self.parts.reverse()
        for part in self.parts:
            part.reverse()

    def rotate(self, pivot_point, rotation_angle):
        """
        :type pivot_point: (float, float)
        :type rotation_angle: float
        """
        for part in self.parts:
            part.rotate(pivot_point, rotation_angle)


class CurveTracingSettings(object):
    def __init__(self, start_percentile=0, end_percentile=1, rotation_angle=0,
                 fixed_angle=False, width_scale=1, fixed_width=False):
        """
        Curve tracing settings is used when you are trying to generate traced points. You can specify lots of options
        like making the curve stick to a certain part of the tracing structure by altering the start and end percentile.

        :param start_percentile: start tracing percentile, doesn't have to be less than end_percentile
        :param end_percentile: end tracing percentile, doesn't have to be greater than start_percentile
        :param rotation_angle: the angle to add to its perpendicular point
        :param fixed_angle: if you hate dynamic angle, change this to True (normally, this is not useful)
        :param width_scale: if negative, the curve is mirror flipped, the axis is its tracing structure's straight line
        :param fixed_width: change to True if you don't want the curve to expand proportionally to its tracing structure
        """
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.rotation_angle = rotation_angle
        self.fixed_angle = fixed_angle
        self.width_scale = width_scale
        self.fixed_width = fixed_width

    @property
    def rotation_angle(self):
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        self._rotation_angle = angle_trunc(value)

    def mirror_flip(self):
        self.width_scale *= -1


class CurveArchetype(object):
    """
    Collection of curves and its tracing structure, the curves are not needed to be connected.
    The tracing structure should not be curvy, the best is a straight line without control points.

    The purpose of curve archetype is to store discontinuous, fragmented, discrete curves as a group.

    :type tracing_structure: ICurve
    :type curves: list[ICurve]
    :type curve_tracing_settings_list: list[CurveTracingSettings]
    """

    def __init__(self, tracing_structure):
        """
        :type tracing_structure: ICurve
        """
        assert isinstance(tracing_structure, ICurve)
        self.tracing_structure = tracing_structure
        self.curves = []
        self.curve_tracing_settings_list = []

    def add_curve(self, curve, cts=None):
        """
        :param curve: the curve to add
        :type curve: ICurve
        :param cts: Curve tracing settings
        :type cts: CurveTracingSettings
        """
        self.curves.append(curve)
        self.curve_tracing_settings_list.append(cts if cts else CurveTracingSettings())

    def mirror_flip(self):
        for cts in self.curve_tracing_settings_list:
            cts.mirror_flip()

    def set_rotation_angle(self, angle):
        for cts in self.curve_tracing_settings_list:
            cts.rotation_angle = angle

    def scale_width(self, scale):
        """
        Scale the current width by 'scale' (multiplication) taking its old scale into account.

        :type scale: float
        """
        for cts in self.curve_tracing_settings_list:
            cts.width_scale *= scale

    def set_width_scale(self, width_scale):
        """
        Set all the current width to the specified 'width_scale' ignoring its old scale.
        :param width_scale: the scale to set
        :type width_scale: float
        """
        for cts in self.curve_tracing_settings_list:
            cts.width_scale = width_scale

    def set_fixed_width(self, fixed_width):
        """
        :type fixed_width: bool
        """
        for cts in self.curve_tracing_settings_list:
            cts.fixed_width = fixed_width

    def set_fixed_angle(self, fixed_angle):
        """
        :type fixed_angle: bool
        """
        for cts in self.curve_tracing_settings_list:
            cts.fixed_angle = fixed_angle

    def get_traced_points_list(self, target_structure=None):
        """
        Generate similar curve points from this CurveArchetype as a list

        :param target_structure: Target tracing structure, defaults to original tracing structure if not provided
        :type target_structure: ICurve

        :return: list of traced points, not a curve instance (you need to choose a way to store it later)
        :rtype: list[(float, float)]
        """
        if target_structure is None:
            target_structure = self.tracing_structure
        traced_points = []
        for i, curve in enumerate(self.curves):
            tps = curve.get_traced_points(self.tracing_structure, target_structure, self.curve_tracing_settings_list[i])
            traced_points.append(tps)
        return traced_points


def get_perpendicular_curve_error(cpoints1, cpoints2, percentiles1, percentiles2):
    """
    :type cpoints1: list[(float, float)]
    :type cpoints2: list[(float, float)]
    :type percentiles1: list[float]
    :type percentiles2: list[float]
    :rtype: float
    """
    assert len(cpoints1) == len(percentiles1) and len(cpoints2) == len(percentiles2)
    if len(cpoints1) > len(cpoints2):
        cpoints1, cpoints2 = cpoints2, cpoints1
        percentiles1, percentiles2 = percentiles2, percentiles1
    if len(cpoints1) - 2 <= 0:
        return 0.0
    error = 0.0
    start_j = 1
    for i in range(1, len(cpoints1) - 1):
        for j in range(start_j, len(cpoints2) - 1):
            if percentiles2[j] > percentiles1[i]:
                previous_point = cpoints2[j - 1]
                next_point = cpoints2[j]
                dist = get_distance(previous_point, next_point)
                angle = get_angle_between_points(previous_point, next_point)
                ratio = (percentiles1[i] - percentiles2[j - 1]) / (percentiles2[j] - percentiles2[j - 1])
                middle_point = get_point_away_from(previous_point, angle, dist * ratio)
                dist_sqr = get_distance_squared(cpoints1[i], middle_point)
                error += dist_sqr
                start_j = j
                break
    return error / (len(cpoints1) - 2)


def get_percentile_list(cpoints):
    """
    :type cpoints: list[(float, float)]
    :rtype: list[float]
    """
    curve_length = 0.0
    percentiles = [0.0]
    for i in range(len(cpoints) - 1):
        curve_length += get_distance(cpoints[i], cpoints[i + 1])
        percentiles.append(curve_length)
    return [p / curve_length for p in percentiles]


def get_curve_length(cpoints):
    """
    :type cpoints: list[(float, float)]
    :rtype: float
    """
    curve_length = 0.0
    for i in range(len(cpoints) - 1):
        curve_length += get_distance(cpoints[i], cpoints[i + 1])
    return curve_length


def get_distance(p1, p2):
    """
    Get absolute distance between 2 points

    :type p1: (float, float)
    :type p2: (float, float)
    :rtype: float
    """
    return get_distance_squared(p1, p2) ** .5


def get_distance_squared(p1, p2):
    """
    Use when you want to compare similarity. This implementation find distribution of points.
    It's like variance, and distance is standard deviation.

    :type p1: (float, float)
    :type p2: (float, float)
    :rtype: float
    """
    x1, y1 = p1
    x2, y2 = p2
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


# total_points excluding start and end points
def get_bezier_points(mpoints, total_points=None, include_start_end=True):
    """
    :type mpoints: list[(float, float)]
    :type total_points: int
    :type include_start_end: bool
    :rtype: list[(float, float)]
    """
    cpoints = []
    if total_points is None:
        total_points = get_total_points(mpoints)
    total_points = int(total_points)
    for i in range(total_points):
        point = get_bezier_point(mpoints, float(i + 1) / (total_points + 1))
        cpoints.append(point)
    if include_start_end:
        cpoints.insert(0, mpoints[0])
        cpoints.append(mpoints[-1])
    return cpoints


def get_total_points(mpoints):
    """
    Get an estimated number of points needed to plot the curve

    :type mpoints: list[(float, float)]
    :rtype: int
    """
    cpoints = []
    for i in range(TOTAL_POINTS + 2):
        t = float(i) / (TOTAL_POINTS + 1)
        point = get_bezier_point(mpoints, t)
        cpoints.append(point)
    estimated_length = get_curve_length(cpoints)
    total_points = estimated_length // LINE_LENGTH  # this is a division not a comment!
    return int(total_points)


def get_bezier_point(mpoints, t):
    """
    :type mpoints: list[(float, float)]
    :type t: float
    :rtype: (float, float)
    """
    if t == 0:
        return mpoints[0]
    if t == 1:
        return mpoints[-1]
    n = len(mpoints) - 1  # degree n
    x, y = 0, 0
    for i in range(n + 1):
        b = get_bernstein_basis_polynomial(i, n, t)
        mx, my = mpoints[i]
        x += b * mx
        y += b * my
    return x, y


def get_bernstein_basis_polynomial(i, n, t):
    """
    Bernstein's Basis Polynomial value

    :type i: int
    :type n: int
    :type t: float
    :rtype: float
    """
    return binomial(n, i) * (t ** i) * (1 - t) ** (n - i)


def binomial(x, y):
    """
    Get a binomial value (number of ways to choose y objects from x objects)

    :param x: n, must be greater than or equal to y
    :type x: int
    :param y: r, must be less than or equal to x
    :type y: int
    :return: binomial value which is usually large, if x < y then return 0
    :rtype: int
    """
    try:
        return factorial(x) // factorial(y) // factorial(x - y)  # the double slash is to divide only in integer form
    except ValueError:
        return 0


def get_perpendicular_point(start, end, p):
    """
    Get the point on a line perpendicular to a chosen point outside the line.

    Start and end are points on the given line.

    :type start: (float, float)
    :type end: (float, float)
    :type p: (float, float)
    :rtype: (float, float)
    """
    x1, y1 = start
    x2, y2 = end
    x3, y3 = p
    px = x2 - x1
    py = y2 - y1
    dab = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(dab)
    x = x1 + u * px
    y = y1 + u * py
    return x, y


def get_middle_point(cpoints):
    """
    Get the middle point from a given list of points

    :type cpoints: list[(float, float)]
    :rtype: (float, float)
    """
    assert cpoints
    x, y = 0.0, 0.0
    for point in cpoints:
        x += point[0]
        y += point[1]
    n = len(cpoints)
    return x / n, y / n


# region Unused get_time_series
# def get_time_series(cpoints, midpoint=None):
#     """
#     :type cpoints: list
#     :type midpoint: tuple
#     :rtype: list
#     """
#     assert cpoints
#     timeseries = []
#     middlepoint = midpoint if midpoint else get_middle_point(cpoints)
#     for point in cpoints:
#         dist = get_distance(point, middlepoint)
#         timeseries.append(dist)
#     return timeseries
# endregion


# region Unused get_dtw_distance
# def get_dtw_distance(ts1, ts2):
#     """Dynamic time warping between time series
#     :rtype: float
#     """
#     n, m = len(ts1), len(ts2)
#     dtw = [[None for _ in range(m + 1)] for __ in range(n + 1)]  # create a 2D array of size n+1, m+1
#     for i in xrange(1, n + 1):
#         dtw[i][0] = float('inf')
#     for i in xrange(1, m + 1):
#         dtw[0][i] = float('inf')
#     dtw[0][0] = 0
#     for i in xrange(1, n + 1):
#         for j in xrange(1, m + 1):
#             cost = abs(ts1[i - 1] - ts2[j - 1])
#             dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
#     return dtw[n][m]
# endregion


# region Unused get_appropriate_t Use to find closest point on a curve to a given point
# def get_appropriate_t(curve, chosen_point):
#     """
#     :rtype: float
#     """
#     total = int(1 / T_PRECISION)
#     chosen_t, least_dist = None, None
#     for i in range(total + 1):
#         t = float(i) / total
#         point = get_bezier_point(curve, t)
#         dist = get_distance(point, chosen_point)
#         if least_dist is None or dist < least_dist:
#             least_dist = dist
#             chosen_t = t
#     # print chosen_t, least_dist
#     return chosen_t
# endregion


def angle_trunc(angle):
    """
    Truncate an angle to make it in range from 0 to 2*pi

    :param angle: angle in radians
    :type angle: float
    :return: truncated angle (measured in radians)
    :rtype: float
    """
    angle %= pi * 2
    return angle


def get_angle_between_points(orig, landmark):
    """
    The angle measurement is always start from 0 at east direction and measured in radians

    :param orig: origin point
    :type orig: (float, float)
    :param landmark: target point
    :type landmark: (float, float)
    :return: an angle from orig to landmark in radians or 0 if orig and landmark are the same point
    :rtype: float
    """
    x_orig, y_orig = orig
    x_landmark, y_landmark = landmark
    delta_y = y_landmark - y_orig
    delta_x = x_landmark - x_orig
    return angle_trunc(atan2(delta_y, delta_x))


def get_point_away_from(point, angle, distance):
    """
    Get a new point at your desired direction

    :param point: start point
    :type point: (float, float)
    :param angle: angle start from 0 at east (measured in radians)
    :type angle: float
    :param distance: distance from the start point to a new point
    :type distance: float
    :return: a new point
    :rtype: (float, float)
    """
    x, y = point
    x += distance * cos(angle)
    y += distance * sin(angle)
    return x, y


def is_approx_equal(a, b, epsilon_tolerance=1e-10):
    """
    Use when comparing number closing to zero or number which should have the same value

    :type a: float
    :type b: float
    :param epsilon_tolerance: an allowable amount of the difference between a and b to be considered approximately equal
    :type epsilon_tolerance: float
    :return: true if a and b are approximately equal, false otherwise
    :rtype: bool
    """
    return abs(a - b) <= epsilon_tolerance


def get_closest_point(points, comparison_point):
    """
    :type points: list[(float, float)]
    :type comparison_point: (float, float)
    :rtype: (float, float)
    """
    return min(points, key=lambda point: get_distance(comparison_point, point))


def get_angle(points, index):
    """
    Get angle of a given point from a point list specified by the index parameter

    :type points: list[(float, float)]
    :type index: int
    :return: angle measured in radians
    :rtype: float
    """
    last_index = len(points) - 1
    assert 0 <= index <= last_index
    if index == 0:
        angle = get_angle_between_points(points[0], points[1])
    elif index == last_index:
        angle = get_angle_between_points(points[-2], points[-1])
    else:
        angle = get_angle_between_points(points[index - 1], points[index + 1])
    return angle_trunc(angle)


def get_angle_list(points):
    """
    Get angle list from a given list of points

    :type points: list[(float, float)]
    :rtype: list[float]
    """
    angle_list = []
    for i in range(len(points)):
        angle_list.append(get_angle(points, i))
    return angle_list


def get_difference_between_angles(minuend, subtrahend):
    """
    Get the difference between 2 angles between -pi to pi

    Use when you want to compare different size of angle difference

    :param minuend: minuend angle
    :type minuend: float
    :param subtrahend: subtrahend angle
    :type subtrahend: float
    :return: a signed angle which its absolute value won't exceed pi (range inclusively from -pi to pi)
    :rtype: float
    """
    difference = minuend - subtrahend
    difference = angle_trunc(difference)
    # do not try to change this code, the angle should not exceed pi ever! if you don't understand, draw a picture!
    # the difference in angle should always be between -pi and pi
    # range between 0 and 2*pi is also the same angle but isn't useful when comparing angle size with another angle
    if difference >= pi:
        difference -= pi * 2
    return difference


def get_angle_derivative(angles, index):
    """Find how much angle changes at a certain point, can be use repeatedly to find 2nd, 3rd or n-th derivative

    :param angles: angle list which is required for differentiation (measured in radians)
    :type angles: list[float]
    :param index: the index on the list to find angle derivative at
    :type index: int
    :return: angle derivative (how much angle changes relative to nearby points)
    :rtype: float
    """
    last_index = len(angles) - 1
    assert 0 <= index <= last_index
    if index == 0:
        derivative = get_difference_between_angles(angles[1], angles[0])
    elif index == last_index:
        derivative = get_difference_between_angles(angles[-1], angles[-2])
    else:
        derivative = get_difference_between_angles(angles[index + 1], angles[index - 1])
    return derivative  # do not use angle_trunc function because it will change the derivative from negative to positive


def get_angle_derivative_list(angles):
    """
    Find how much angle changes at a certain point, can be use repeatedly to find 2nd, 3rd or n-th order derivative

    :param angles: angle list which is required for differentiation (measured in radians)
    :type angles: list[float]
    :return: list of angle derivative (how much angle changes relative to nearby points)
    :rtype: list[float]
    """
    assert isinstance(angles, list)
    derivative_list = []
    for i in range(len(angles)):
        derivative_list.append(get_angle_derivative(angles, i))
    return derivative_list


def get_angles_mean(angle_list, distance_list=None):
    """
    Find the mean angle of all angles in a list

    :param angle_list: list of angle (measured in radians)
    :type angle_list: list[float]
    :param distance_list: the distance of each angle, default to 1, which means all angle has the same priority
    :type distance_list: list[float]
    :return: mean angle (measured in radians)
    :rtype: float
    """
    total_angle = len(angle_list)
    assert total_angle > 0
    if distance_list is None:
        distance_list = [1] * total_angle
    assert len(distance_list) == total_angle
    sin_sum = 0
    cos_sum = 0
    for i in range(total_angle):
        angle = angle_list[i]
        distance = distance_list[i]
        sin_sum += sin(angle) * distance
        cos_sum += cos(angle) * distance
    return atan2(sin_sum / total_angle, cos_sum / total_angle)


def get_total_peaks(datas, max_data=pi, sd_factor=1):
    """
    Get total number of peak values in data list including both top peaks and bottom peaks

    This algorithm is not quite good enough. There is some constants that you can change to make it work better.

    :param datas: data list as type int or float
    :type datas: list[float]
    :param max_data: maximum data threshold
    :type max_data: float
    :param sd_factor: minimum factor value that will tell a data is considered a peak value.
        The higher, the lower number of peaks returned
    :type sd_factor: float
    :return: total number of peaks gathered
    :rtype: float
    """
    if not datas:
        return 0
    datas = [abs(data) for data in datas]  # this will not alter the original datas list
    mean = numpy.mean(datas)
    sd = numpy.std(datas)
    # print 'mean:', degrees(mean)
    # print 'SD:', degrees(sd)
    allowed_peak_threshold = mean + sd * sd_factor
    # print 'allowed_peak_threshold:', degrees(allowed_peak_threshold)
    multiplier = 1  # 4 * (mean / max_data)  # this constant is to make sure that the large data = more peaks
    total_peaks = int(sum(1 for data in datas if data > allowed_peak_threshold) * multiplier)
    return total_peaks


def check_percentile(percentile):
    """
    :type percentile: float
    :rtype: int
    """
    if is_approx_equal(percentile, 0):
        return 0
    if is_approx_equal(percentile, 1):
        return 1
    if not 0 <= percentile <= 1:
        raise ValueError('percentile should be in range 0..1, current value is %f' % percentile)
    return percentile


def get_random_point_around(point, max_radius=30):
    return get_point_away_from(point, random.TWOPI * random.random(), random.uniform(0, max_radius))


def get_curve_points_near_point(curve_points, point):
    curve_to_point_distance = lambda cpoints: get_distance(get_closest_point(cpoints, point), point)
    return min(curve_points, key=curve_to_point_distance)
