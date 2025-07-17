from datetime import datetime
import math
import logging
from typing import List, Type, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.project2.speed_DP_decider import SPEED_DP_DECIDER_v1
from nuplan.planning.simulation.planner.project2.speed_QP_planner import QP_SPEED_PLANNER_v1
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.planner.project2.path_QP_planner import QP_PATH_PLANNER_v1
from nuplan.planning.simulation.planner.project2.my_predictor import MyPredictor_v1

logger = logging.getLogger(__name__)


class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 10.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6)) # 把规划时域从秒->微秒（TimePoint类要求）
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None # ：Optional[class]表示_router变量既可以是一个BFSRouter类型变量，也可以是None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False


    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observation = current_input.history.current_state
        ego_state_list = current_input.history.ego_states
        observation_list= current_input.history.observations
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line # TODO:根据平滑后的结果对从router查出来的lb和rb进行修正
        self._reference_path_provider = ReferenceLineProvider(self._router) # 映入二次规划平滑参考线
        self._reference_path_provider._reference_line_generate(ego_state) 

        # 3. Objects prediction
        self._predictor = MyPredictor_v1(ego_state, observation, ego_state_list, observation_list, self.horizon_time, self.sampling_time)
        vehicles = self._predictor.predict()
        

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, vehicles, 
                                                    self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)
    
    # TODO: 2. Please implement your own trajectory planning.
    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 objects: List[TrackedObject],
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        return: trajectory Planning result
        """

        # 1.Path planning
        # 1.1 path qp with piece_wise_jerk
        path_planner = QP_PATH_PLANNER_v1(ego_state, reference_path_provider) # set planner
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = path_planner.planning_on_frenet() # QP planning

        # 2.Transform path planning result to cartesian frame
        path_idx2s, path_x, path_y, path_heading, path_kappa = transform_path_planning(optimal_path_s, optimal_path_l, \
                                                                                       optimal_path_dl,
                                                                                       optimal_path_ddl, \
                                                                                       reference_path_provider)

        # 3.Speed planning
        # 3.1 get planning start state
        start_s, start_l, start_s_dot, start_l_dot, \
            start_dl, start_l_dot2, start_s_dot2, start_ddl = path_planner.generate_planning_start_state()
        # 3.2 get max_velocity from router
        # 检查是否有router里的max_vel，很可能没有！！
        max_velocity_from_router = path_planner.get_router_max_velocity(start_s) 
        max_velocity = max_velocity_from_router if not math.isnan(max_velocity_from_router) else self.max_velocity
        # 3.3 speed decider with dp
        dp_step = 0.5 # 设置dp的t采样间隔，smapling_time=0.25搜索规模太大了 
        speed_dp_decider = SPEED_DP_DECIDER_v1(ego_state, objects, \
                                               path_idx2s, path_x, path_y, path_heading, path_kappa,\
                                                self.horizon_time.time_s, dp_step, 
                                                max_velocity, ego_v=start_s_dot, max_acc=4, max_dec=-6)
        s_lb, s_ub, dp_s_out, dp_t_out, dp_st_s_dot = speed_dp_decider.dynamic_programming()
        # 3.4 speed qp with picec wise jerk 
        min_velocity = 0
        reference_velocity = max_velocity 
        speed_qp_planner = QP_SPEED_PLANNER_v1(ego_state, start_s, start_s_dot, start_s_dot2, \
                                 s_lb, s_ub, dp_s_out, dp_t_out,
                                max_velocity, min_velocity, reference_velocity)
        optimal_speed_s, optimal_speed_s_dot, \
            optimal_speed_s_2dot, optimal_speed_t = speed_qp_planner.speed_qp_planning()

        # 4.Produce ego trajectory
        # 4.1 generate cur state
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        plot_base_x = ego_state.center.x
        plot_base_y = ego_state.center.y
        # 4.2 generate traj in planning_t
        trajectory: List[EgoState] = [state]
        print(f"ego_state在time_point={ego_state.time_point.time_s}[s]时的规划完成：")
        t_end = optimal_speed_t[-1] # dp时候不一定t_end=horizen_time，所以这里修改demo代码
        plt.clf()
        for iter in range(int(t_end / sampling_time.time_s)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            # 插值是因为QP_speed的点间距不一定是sampling_time，所以如果按sample去插值轨迹可能导致查询relative_time不在optimal_speed_t内
            s, velocity, accelerate = cal_dynamic_state(relative_time, optimal_speed_t, optimal_speed_s,
                                                        optimal_speed_s_dot, optimal_speed_s_2dot)
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            # note:由于插值需要查询值在x范围内，速度规划时按max=20，horizen=8来算，s最多到160m，所以path planning时也要给出至少160m
            # 但在speed_dp为了提高速度只规划80m，按理来说可以减小path_QP的size，不过没有必要因为本来这里设置的就比较小
            x, y, heading, _ = cal_pose(s, path_idx2s, path_x, path_y, path_heading, path_kappa)


            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(velocity, 0),
                rear_axle_acceleration_2d=StateVector2D(accelerate, 0),
                tire_steering_angle=heading,
                time_point=state.time_point + sampling_time,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0,
                angular_accel=0,
            )
            print(f"x={state.center.x},y={state.center.y},vx={state.dynamic_car_state.rear_axle_velocity_2d.x},vy={state.dynamic_car_state.rear_axle_velocity_2d.y} ")
            trajectory.append(state)
            plt.plot(state.center.x - plot_base_x, state.center.y - plot_base_y, color='red')
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        file_name_prefix = "/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/test_fig_traj_XY"
        file_extension = ".png"
        file_name = f"{file_name_prefix}/trajXY_{formatted_datetime}{file_extension}"
        plt.savefig(file_name)
        plt.close()
        print("--------------------------------------------------------------------------")


        return trajectory
    

