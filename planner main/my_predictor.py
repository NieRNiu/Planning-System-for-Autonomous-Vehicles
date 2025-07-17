import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from collections import deque

from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory

class MyPredictor_v1(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observation: Observation, 
                 ego_state_list: list[EgoState],
                 observation_list: list[Observation],
                 duration: TimePoint, sample_time: TimePoint,
                 occupancy_map_radius=30) -> None:
        '''
        :param: ego_state
        :param: observation
        :param: ego_state_buffer
        :param: observation_buffer
        :param: duration 
        :param: sample_time
        :param: occupancy_map_radius 感兴趣的区域半径,只考虑该区域内的agent default=40[m]
        '''
        self._ego_state = ego_state
        self._observation = observation
        self._ego_state_list = ego_state_list
        self._ego_observation_lis = observation_list
        self._duration = duration
        self._sample_time = sample_time
        self._occupancy_map_radius = occupancy_map_radius

    def get_single_obstacles_info(self, cur_observation: Observation):
        '''
        针对某一时刻的observation, 按照类别抽取objects, 把动态vehcile单独分为一类, 其他分为一类
        :param: cur_observation 某一时刻的observation
        :return obstacles, vehicles 
        '''
        if isinstance(cur_observation, DetectionsTracks):
            # 筛选处几个的种类的障碍物 
            object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                            TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                            TrackedObjectType.GENERIC_OBJECT]
            objects = cur_observation.tracked_objects.get_tracked_objects_of_types(object_types)

            obstacles = []
            vehicles = []
            for obj in objects:
                if obj.box.geometry.distance(self._ego_state.car_footprint.geometry) > self._occupancy_map_radius: # 找object的几何框与自车几何框的距离，
                # if np.linalg.norm(self._ego_state.center.array - object.center.array) < self._occupancy_map_radius # 找object中心与自车中心的距离
                    continue
                # 动态车辆单独归一类，其他归为obstacles, 路径规划时单独提取静态障碍物投影，速度规划直接用预测信息
                if obj.tracked_object_type == TrackedObjectType.VEHICLE: 
                    if obj.velocity.magnitude() < 0.01:
                        obstacles.append(obj)
                    else:
                        vehicles.append(obj)
                else:
                    obstacles.append(obj)
        else:
            raise ValueError(
                f"SimplePredictor only supports DetectionsTracks. Got {self._observation.detection_type()}")
        return obstacles, vehicles 
    
    def get_vehcile_prediction_with_constant_velocity(self, vehicle: TrackedObject):
        '''
        对单个车辆进行定速度预测
        :param: vehcile 单个车辆object对象
        :return: vehcile_predicted_traj: list[WaypointTypes = Union[Waypoint, EgoState]]
        '''
        # 解包当前信息用于预测
        cur_x = vehicle.center.x
        cur_y = vehicle.center.y
        cur_vx = vehicle.velocity.x
        cur_vy = vehicle.velocity.y
        cur_heading = vehicle.box.center.heading
        length = vehicle.box.length
        width = vehicle.box.width
        height = vehicle.box.height
        # 从自车当前状态获取时间戳
        cur_time_point = self._ego_state.time_point 
        waypoint = Waypoint(time_point=cur_time_point, \
                                oriented_box=OrientedBox(center=vehicle.center, \
                                        length=length, width=width, height=height), \
                                velocity=vehicle.velocity
                                )
        
        waypoints: List[Waypoint] = [waypoint]
        pred_traj: List[PredictedTrajectory] = []
        for iter in range(int(self._duration.time_us / self._sample_time.time_us)):
            relative_time = (iter + 1) * self._sample_time.time_s # 计算相对时间
            pred_x = cur_x + cur_vx * relative_time  # 匀速推x位置
            pred_y = cur_y + cur_vy * relative_time  # 匀速推y位置
            pred_velocity = StateVector2D(cur_vx, cur_vy) # 匀速速度不变
            # 修改orientedBox的center变量
            pred_orientedbox = OrientedBox(center=StateSE2(pred_x, pred_y, cur_heading),
                                           length=length, width=width, height=height)
            waypoint = Waypoint(time_point=waypoint.time_point + self._sample_time, 
                                oriented_box=pred_orientedbox,
                                velocity=pred_velocity)
            waypoints.append(waypoint)
        
        pred_traj.append(PredictedTrajectory(probability=1.0, waypoints=waypoints))
        
        return pred_traj
    
    def get_obstalce_prediction_with_constant_velocity(self, obstacle: TrackedObject):
        '''
        对单个障碍物进行定速度预测
        :param: obstacle 单个object对象
        :return: obstacle_predicted_traj: list[WaypointTypes = Union[Waypoint, EgoState]]
        '''
        return []

    

    def predict(self):
        """Inherited, see superclass.
        :return: vehciles List[PredictedTrajectory]
        :return: obstalces List[PredictedTrajectory](暂时不return该值,先用vehicle的预测做)
        """
        # get cur objects info 
        #  解包当前时刻的观测
        obstacles, vehicles = self.get_single_obstacles_info(self._observation)
        # TODO: 添加obstacle的预测
        # 添加车辆预测
        for vehicle in vehicles:
            vehicle_predicted_traj = self.get_vehcile_prediction_with_constant_velocity(vehicle)
            vehicle.predictions = vehicle_predicted_traj
        # # 添加障碍物预测
        # for obstacle in obstacles:
        #     obstacle_predicted_traj = self.get_obstalce_prediction_with_constant_velocity(obstacle)
        #     obstacle.predictions = obstacle_predicted_traj
        return vehicles