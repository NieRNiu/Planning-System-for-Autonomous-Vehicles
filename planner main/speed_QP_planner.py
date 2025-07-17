from datetime import datetime
import math
import logging
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
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
from nuplan.planning.simulation.planner.project2.frame_transform import get_match_point, cal_project_point, cartesian2frenet
import matplotlib.pyplot as plt

import cvxpy as cp
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

class QP_SPEED_PLANNER_v1:
    '''

    '''
    def __init__(self, 
                ego_state: EgoState,
                plan_start_s: float,
                plan_start_s_dot: float,
                plan_start_s_dot2: float,
                s_lb: List[float],
                s_ub: List[float],
                dp_s_out: List[float],
                dp_t_out: List[float],
                max_vel: float,
                min_vel: float,
                reference_vel: float
                ) -> None:
        self._ego_state = ego_state
        self._plan_start_s = plan_start_s
        self._plan_start_s_dot = plan_start_s_dot
        self._plan_start_s_dot2 = plan_start_s_dot2
        self._s_lb = s_lb
        self._s_ub = s_ub
        self._dp_s = dp_s_out
        self._dp_t = dp_t_out
        self._max_vel = max_vel
        self._min_vel = min_vel
        self._reference_vel = reference_vel

    def speed_qp_planning(self, w_cost_s_dot2=10, w_cost_jerk=50, w_cost_v_ref=500, k=0.2):
        # 速度dp时，由于speed_curve_end不一定在右边界（最后一列），所以不一定采到了horizen_time
        # dp时初始化dp_s为-1，size是t_list(不包括t=0）根据此找到实际的最后一个s，记录idx
        dp_speed_end_idx = len(self._dp_s) - 1
        for i in range(len(self._dp_s) - 1, -1, -1): # 从后往前遍历，遍历到-1)，即0,
            if self._dp_s[i] != -1:
                dp_speed_end_idx = i
                break
        
        n = dp_speed_end_idx + 1 + 1 # 实际需要优化的点的个数   = 最后一个非-1s的索引 + 1 + 开头点(0, 0)
        dt = self._dp_t[1] - self._dp_t[0] # 采样间隔（优化点t间隔）
        # t_end = self._dp_t[dp_speed_end_idx] # 实际dp出来的时间，也是本轮planning实际的horizon_time

        # 等式约束矩阵 Aeq * x = beq -------------------------------------------------------------
        Aeq_sub = np.array([[1, dt, dt**2/3, -1, 0, dt**2/6], 
                            [0, 1, dt/2, 0, -1, dt/2]])
        Aeq = np.zeros((2*n - 2, 3*n)) 
        Beq = np.zeros(2*n - 2)   # 在cvxpy中，定义的变量x = cp.Variable(3*n)是一个向量，等同于NumPy中的一维数组（形状为(3*n,)），所以这里不应该是(2*n-2, 1)
        # 填充Aeq矩阵
        for i in range(n - 1):  # i = 0 : n - 2
            row = 2*i 
            col = 3*i 
            Aeq[row : row + 2, col : col + 6] = Aeq_sub # 注意切片操作的左闭右开
        
        # 不等式约束矩阵 Ax <= b -------------------------------------------------------------
        # 不允许倒车 s(i) - s(i+1) <= 0
        A = np.zeros((n - 1, 3 * n))
        b = np.zeros(n - 1)
        for i in range(n - 1):
            A[i, 3 * i] = 1
            A[i, 3 * i + 3] = - 1
            
        # 优化变量上下界约束 -----------------------------------------------------------------------
        # 初始化 lb 和 ub
        lb = np.ones(3*n) * np.nan  # 使用 np.nan 初始化，以便之后可以识别哪些值被更新
        ub = np.ones(3*n) * np.nan
        # 更新 lb 和 ub 的值，对除去起始点(0,0)之外的点做约束
        # qp优化时在开头多加了一个点(0,0)，该点在dp的结果里没有给出上下界，所以在索引s_lb和s_ub时索引要-1
        for i in range(1, n):  # Python中索引从0开始
            # 允许最小加速度为-6 最大加速度为4(基于车辆动力学)
            lb[3*i] = self._s_lb[i - 1]
            lb[3*i + 1] = self._min_vel
            lb[3*i + 2] = -6
            ub[3*i] = self._s_ub[i - 1]
            ub[3*i + 1] = self._max_vel
            ub[3*i + 2] = 4
        # 起点约束,给k的宽容度
        lb[0] = 0
        # 预测模型太简单，可能有第二个点（0+dt，s）的s必须为0的情况，或者说必须急刹
        lb[1] = self._plan_start_s_dot - abs(self._plan_start_s_dot * k) \
            if self._plan_start_s_dot - abs(self._plan_start_s_dot * k) < 0 else 0 
        lb[2] = self._plan_start_s_dot2 - abs(self._plan_start_s_dot2 * k) \
            if self._plan_start_s_dot2 - abs(self._plan_start_s_dot2 * k) < 0 else 0
        ub[0] = 0
        ub[1] = self._plan_start_s_dot + abs(self._plan_start_s_dot * k)
        ub[2] = self._plan_start_s_dot2 + abs(self._plan_start_s_dot2 * k)
        
        # 代价函数 -----------------------------------------------------------------------
        A_s_dot2 = np.zeros((3*n, 3*n))
        A_jerk = np.zeros((n - 1, 3*n))
        A_ref = np.zeros((3*n, 3*n))
        # 填充
        for i in range(1, n):
            A_s_dot2[3*i + 2, 3*i + 2] = 1 #(2,2) (5, 5) ..... (3n-1, 3n-1) 
            A_ref[3*i + 1, 3*i + 1] = 1 # (1,1) (4,4) .......(3n-2, 3n-2)
        A_sub = np.array([0, 0, 1, 0, 0, -1])
        for i in range(n - 1):
            A_jerk[i, 3*i : 3*i + 6] = A_sub
        # 生成H
        H = w_cost_s_dot2 * np.dot(A_s_dot2, A_s_dot2.T) + \
            w_cost_jerk * np.dot(A_jerk.T, A_jerk) + \
            w_cost_v_ref * np.dot(A_ref, A_ref.T)
        H = 2 * H

        # 生成f -----------------------------------------------------------------------
        f = np.zeros((3*n, 1))
        for i in range(n):
            f[3*i + 1] = -2 * w_cost_v_ref * self._reference_vel
        
        # 求解二次规划问题 ----------------------------------------------
        x = cp.Variable(3*n)  # 决策变量
        objective = cp.Minimize(0.5 * cp.quad_form(x, H) + f.T @ x) # 目标函数
        constraints = [A @ x <= b, Aeq @ x == Beq, x >= lb, x <= ub] # 约束条件
        # 定义优化问题
        prob = cp.Problem(objective, constraints)
        # 求解优化问题
        result = prob.solve(solver=cp.OSQP) # 指定求解器
        # 初始化结果列表
        qp_speed_s = []
        qp_speed_ds = []
        qp_speed_dds = []
        qp_speed_t = []
        # 检查问题是否成功解决
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            print("speed QP Problem solved successfully.")
            for i in range(n):
                qp_speed_s.append(x.value[3*i])     # 提取位置
                qp_speed_ds.append(x.value[3*i + 1]) # 提取速度
                qp_speed_dds.append(x.value[3*i + 2])# 提取加速度
                qp_speed_t.append(i * dt)
            plt.plot(qp_speed_t, qp_speed_s, color='green')
            # 返回路径规划结果
            # print("qp_path_l:", qp_speed_s)
            # print("qp_path_dl:", qp_speed_ds)
            # print("qp_path_ddl:", qp_speed_dds)
            # TODO: PLOT
        else:
        # if(True):
            print("speed QP Problem solving failed with status:", prob.status) # 这里崩溃的一大原因就是plan_start_s_dot < 0
            # √ TODO: 给一个规划失败后的备用方案,让仿真不崩溃
            # 方案一：如果优化失败，就沿用DP的结果，拟合
            # dp_s_new = []
            # dp_s_new.append(0)
            # dp_t_new = []
            # dp_t_new.append(0)
            # for i in range(n - 1):
            #     dp_s_new.append(self._dp_s[i])
            #     dp_t_new.append(self._dp_t[i])
            # # 拟合
            # degree = 5  # 多项式的度数
            # coeffs = np.polyfit(dp_t_new, dp_s_new, degree)
            # poly = np.poly1d(coeffs)
            # qp_speed_s = poly(dp_t_new).tolist()
            # qp_speed_ds = np.polyder(poly, 1)(dp_t_new).tolist()  # 一阶导数
            # qp_speed_dds = np.polyder(poly, 2)(dp_t_new).tolist()  # 二阶导数
            # qp_speed_t = dp_t_new.copy()
            # 方案二：搞一个T秒内匀减速到0的轨迹
            # T_brake = 4.0  # 制动时间
            # dt = 0.5       # 时间步长
            # N = int(T_brake / dt) + 1  # 新的点数，包括起点在内
            # qp_speed_t = [(i - 1) * dt for i in range(1, N + 1)]
            # dds = (0 - self._plan_start_s_dot) / T_brake  # 加速度
            # qp_speed_s = [0]  # 初始位置
            # qp_speed_ds = [self._plan_start_s_dot]  # 初始速度
            # qp_speed_dds = [self._plan_start_s_dot2]  # 加速度保持不变
            # # 使用前一个时刻的速度和位置计算新的位置
            # for i in range(1, N):
            #     new_speed = qp_speed_ds[-1] + dds * dt
            #     new_position = qp_speed_s[-1] + qp_speed_ds[-1] * dt + 0.5 * dds * dt**2
            #     qp_speed_ds.append(new_speed)
            #     qp_speed_s.append(new_position)
            #     qp_speed_dds.append(dds)
            # 方案三： 匀速直线满8s
            T_brake = 8  
            dt = 0.5       # 时间步长
            N = int(T_brake / dt) + 1  # 新的点数，包括起点在内
            qp_speed_t = [(i - 1) * dt for i in range(1, N + 1)]
            dds = 0  # 加速度
            qp_speed_s = [0]  # 初始位置
            if(self._plan_start_s_dot < 0): # 处理本来是倒车的（可能是数据有一些问题啥的）
                self._plan_start_s_dot = 0
            qp_speed_ds = [self._plan_start_s_dot]  # 初始速度
            qp_speed_dds = [self._plan_start_s_dot2]  # 第一个点用起始加速度
            # 使用前一个时刻的速度和位置计算新的位置
            for i in range(1, N):
                new_speed = self._plan_start_s_dot
                new_position = qp_speed_s[-1] + qp_speed_ds[-1] * dt + 0.5 * dds * dt**2
                qp_speed_ds.append(new_speed)
                qp_speed_s.append(new_position)
                qp_speed_dds.append(dds)
            plt.plot(qp_speed_t, qp_speed_s, color='grey') 

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        file_name_prefix = "/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/test_fig_ST_QP"
        file_extension = ".png"
        file_name = f"{file_name_prefix}/STQPmap_{formatted_datetime}{file_extension}"
        plt.savefig(file_name)
        plt.close()

        # note: 虽然约束了qp_speed_s[0]=0，但是求解出来可能出现一个非常小(-1e05)这种的值，所以这里约束一下避免后续插值报错：
        qp_speed_s[0] = 0
        return qp_speed_s, qp_speed_ds, qp_speed_dds, qp_speed_t

        



        
        