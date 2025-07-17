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
from nuplan.planning.simulation.planner.project2.frame_transform import get_match_point, \
    cal_project_point, cartesian2frenet, local2global_vector
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import cvxpy as cp


class QP_PATH_PLANNER_v1:
    '''
    v1: 不考虑避障,仅做上下界内的轨迹规划
    :param ego_state current state of ego
    :param reference_line_provider
    :param planning_len  前向规划路径长度 default = 80[m] 约定 planning_len < 200 (参考线只前向取200)
    :param planning_n QP优化的点的个数 default = 40 约定 plnning_len / planning_n >= 1 且为整数
    :param n_final 对QP优化点进行增密后的轨迹点个数 default = 401
    '''
    def __init__(self, 
                ego_state: EgoState,
                reference_line_provider: ReferenceLineProvider,
                planning_len = 80,
                planning_n = 40,
                n_final = 401,
                ) -> None:
        # 第一次使用nuplan，先把解包数据统一写在init，学习api
        # 解包所需自车信息
        
        # test_vx_1 = ego_state.dynamic_car_state.center_velocity_2d.x # 8.948770512241957
        # test_vy_1 = ego_state.dynamic_car_state.center_velocity_2d.y # -0.08279994027509457
        # test_x_1 = ego_state.center.x
        # test_x_2 = ego_state.center.y
        # test_vx_2 = ego_state.dynamic_car_state.rear_axle_velocity_2d.x # 8.948770512241957
        # test_vy_2 = ego_state.dynamic_car_state.rear_axle_velocity_2d.y # -0.08279994027509457
        # test_vx_3 = ego_state.waypoint.velocity.x # 8.948770512241957
        # test_vy_3 = ego_state.waypoint.velocity.y # -0.08279994027509457
        # test_x_3 = ego_state.waypoint.center.x
        # test_y_3 = ego_state.waypoint.center.y

        # 状态信息，规划一般用后轴中心
        self._ego_x = ego_state.rear_axle.x
        self._ego_y = ego_state.rear_axle.y
        self._ego_vx_local = ego_state.dynamic_car_state.rear_axle_velocity_2d.x # _local车身坐标系
        self._ego_vy_local = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        self._ego_ax_local = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        self._ego_ay_local = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        self._ego_heading = ego_state.rear_axle.heading
        # 几何信息
        self._vehicle_front_length = ego_state.car_footprint.vehicle_parameters.front_length # dist. from rear axle to front bumper
        self._vehicle_rear_length = ego_state.car_footprint.vehicle_parameters.rear_length #  dist. from rear axle to rear bumpe
        self._vehicle_wheel_base = ego_state.car_footprint.vehicle_parameters.wheel_base
        self._vehicle_half_width = ego_state.car_footprint.vehicle_parameters.half_width
        # 解包所需参考线信息
        self._reference_line_provider = reference_line_provider
        self._s_of_reference_line = reference_line_provider._s_of_reference_line
        self._x_of_reference_line = reference_line_provider._x_of_reference_line
        self._y_of_reference_line = reference_line_provider._y_of_reference_line
        self._heading_of_reference_line = reference_line_provider._heading_of_reference_line
        self._kappa_of_reference_line = reference_line_provider._kappa_of_reference_line
        self._lb_of_reference_line = reference_line_provider._lb_of_reference_line
        self._rb_of_reference_line  = reference_line_provider._rb_of_reference_line 
        self._max_v_of_reference_line = reference_line_provider._max_v_of_reference_line
        # 规划信息
        self._n = planning_n
        self._planning_len = planning_len
        self._n_final = n_final
    
    def generate_planning_start_state(self):
        '''
        通用方法，计算车辆当前状态在参考线上的投影,作为planning的start_state
        return start_s, start_l, start_s_dot, start_l_dot, start_dl, start_l_dot2, start_s_dot2, start_ddl
        '''
        # 车身转全局
        ego_vx_global, ego_vy_global = local2global_vector(self._ego_vx_local, self._ego_vy_local,\
                                                           self._ego_heading)
        ego_ax_global, ego_ay_global = local2global_vector(self._ego_ax_local, self._ego_ay_local,\
                                                           self._ego_heading)
        # 找投影点
        start_s, start_l, start_s_dot, start_l_dot,\
              start_dl, start_l_dot2, \
                start_s_dot2, start_ddl = cartesian2frenet([self._ego_x], [self._ego_y], \
                                                            [ego_vx_global], [ego_vy_global],
                                                            [ego_ax_global], [ego_ay_global],
                                                            self._x_of_reference_line, self._y_of_reference_line, 
                                                            self._heading_of_reference_line, self._kappa_of_reference_line, 
                                                            self._s_of_reference_line)
        if(start_s_dot[0] < 0):
            print('debug: start_s_dot[0] < 0,在倒车？')
        return start_s[0], start_l[0], start_s_dot[0], start_l_dot[0], start_dl[0], start_l_dot2[0], start_s_dot2[0], start_ddl[0]
    
    def get_router_max_velocity(self, s):
        '''
        找当前车辆对应参考线上的最大速度
        :param: 查询的车辆frenet下的s
        :return: max_v
        '''
        f_s2v = interp1d(self._s_of_reference_line, self._max_v_of_reference_line)
        max_v = f_s2v(s)
        return max_v

    def generate_convex_space(self):
        '''
        生成凸空间,约束点个数为self._n
        :return l_max: list[float], l_min: list[float]
        '''
        # TODO:考虑对低速障碍物的决策
        # 参考线生成时按ds=1采样，第一个点s=0，现在想往前规划120m，60个，点则每2m一个点，包括起始点
        # 由于第一个点为起始位置（ego当前位置），所以实际规划118m，需要对给出的参考线做裁剪
        step = int(self._planning_len / self._n / 1) 
        l_max = [float(x) for x in self._lb_of_reference_line[0: int(step * self._n - 1) : int(step)]] # note: 要转int，切片操作不允许float作为步长
        l_min = [float(-x) for x in self._rb_of_reference_line[0: int(step * self._n - 1): int(step)]]

        return l_max, l_min
    
    def path_planning_QP(self, plan_start_s: float, plan_start_l: float, plan_start_dl: float, plan_start_ddl: float, 
                              l_min: list[float], l_max: list[float], 
                              w_cost_l=50, w_cost_dl=20000, w_cost_ddl=50, w_cost_dddl=20, w_cost_centre=0, 
                              w_cost_end_l=15, w_cost_end_dl=15, w_cost_end_ddl=15, 
                              delta_dl_max=0.05, delta_ddl_max=0.005,
                              delta_l_max = 0.0, delta_l_min = 0.0,
                              start_constrain_k = 0.0
                              ):
        '''
        piece_wise_jerk算法做路径规划
                    0.5*x'Hx + f'*x = min
                    subject to A*x <= b
                            Aeq*x = beq
                            lb <= x <= ub;
        :param start_planning_state: lan_start_s, plan_start_l, plan_start_dl, plan_start_ddl, 
        :param l_min l_max 凸空间上下界,个数为优化点的数
        :param w_cost_l 参考线代价 default=50
        :param w_cost_dl ddl dddl 光滑性代价 default=50, 20000, 50, 20
        :param w_cost_centre 凸空间中央代价 default=10
        :param w_cost_end_l dl dd1 终点的状态代价 (希望path的终点状态为(0,0,0)) default=15, ,15, 15
        :param delta_dl_max, delta_ddl_max 相邻path_point的dl、ddl的增量限制代价, default=0.5, 0.05
        :param delta_l_max, delta_l_min 对上下边界做拓展,相当于放宽车道边界约束 default=0.0, 0.0
        :param start_constrain_k 对起始位置约束进行放宽(即优化的第一个点可以在规划起始位置上下有一定浮动), defalut=0.2
        '''
        ds = self._planning_len / self._n
        n = self._n

        # 测试参数
        # plan_start_s = 0.0
        # plan_start_l = 0.8
        # plan_start_dl = -1.6
        # plan_start_ddl = -0.02
        # l_min = [-1.43 for _ in range(n)]
        # l_max = [1.31 for _ in range(n)]
        # delta_dl_max = 0.5
        # delta_ddl_max = 0.05
        l_max = [x + delta_l_max for x in l_max]
        l_min = [x - delta_l_min for x in l_min]

        # 计算几何参数用于边界不等式约束
        d1 = self._vehicle_front_length  # distance from rear axle to front_bumper
        d2 = self._vehicle_rear_length # distance from rear axle to rear bumper
        half_width = self._vehicle_half_width

        # 等式约束矩阵生成 Aeq * x = Beq -------------------------------------------------
        # 1. piece_wise_jerk，常三阶导连接相邻点构成等式约束
        Aeq_sub = np.array([[1, ds, ds**2/3, -1, 0, ds**2/6], 
                            [0, 1, ds/2, 0, -1, ds/2]])
        Aeq = np.zeros((2*n - 2, 3*n)) 
        Beq = np.zeros(2*n - 2)   # 在cvxpy中，定义的变量x = cp.Variable(3*n)是一个向量，等同于NumPy中的一维数组（形状为(3*n,)），所以这里不应该是(2*n-2, 1)
        # 填充Aeq矩阵
        for i in range(n - 1):  # i = 0 : n - 2
            row = 2*i 
            col = 3*i 
            Aeq[row : row + 2, col : col + 6] = Aeq_sub # 注意切片操作的左闭右开

        # 不等式约束生成 Ax <= b ------------------------------------------------------
        # 1. 上下限边界约束
        # 边界约束会对起始点的l、dl进行约束，QP这里又固定了起始点位置，因此如果起始位置偏离参考线比较多，路窄、车大会出问题
        # 所以这里考虑不对起始点的边界做约束
        A_sub = np.array([[1, d1, 0],
                        [1, d1, 0],
                        [1, -d2, 0],
                        [1, -d2, 0],
                        [-1, -d1, 0],
                        [-1, -d1, 0],
                        [-1, d2, 0],
                        [-1, d2, 0]])
        A = np.zeros((8*n, 3*n)) # x.form -> (3*n)
        b = np.zeros(8*n)
        # 填充A矩阵
        for i in range(1, n):
            row = 8*i
            col = 3*i
            A[row : row + 8, col : col + 3] = A_sub
        # 生成b矩阵
        front_index = int(np.ceil(d1 / ds)) 
        back_index = int(np.ceil(d2 / ds))
        # 填充b向量
        for i in range(0, n): 
            index1 = min(i + front_index, n - 1) # min避免超出索引个数
            index2 = max(i - back_index, 0)    # max避免出现索引负数
            row = 8 * i
            b[row:row+8] = [
                l_max[index1] - half_width,
                l_max[index1] + half_width,
                l_max[index2] - half_width,
                l_max[index2] + half_width,
                -l_min[index1] + half_width,
                -l_min[index1] - half_width,
                -l_min[index2] + half_width,
                -l_min[index2] - half_width,
            ]
        # 2. 相邻点delat_dl, delat_ddl约束（平滑约束）
        A_dl_minus = np.zeros((n - 1, 3 * n))
        b_dl_minus = np.zeros(n - 1)
        A_ddl_minus = np.zeros((n - 1, 3 * n))
        b_ddl_minus = np.zeros(n - 1)
        # 填充矩阵
        for i in range(n - 1):
            col = 3 * i
            A_dl_minus[i, col:col+6] = [0, -1, 0, 0, 1, 0]
            b_dl_minus[i] = delta_dl_max
            A_ddl_minus[i, col:col+6] = [0, 0, -1, 0, 0, 1]
            b_ddl_minus[i] = delta_ddl_max
        # % -max < a*x < max => ax < max && -ax < -(-max) 拼接矩阵
        A_minus = np.concatenate((A_dl_minus, -A_dl_minus, A_ddl_minus, -A_ddl_minus))
        b_minus = np.concatenate((b_dl_minus, b_dl_minus, b_ddl_minus, b_ddl_minus))
        # 3. 拼接
        A_total = np.concatenate((A, A_minus))
        b_total = np.concatenate((b, b_minus))

        # 优化的上下边界生成（主要是为了约束起始点与planning_start_state不偏离太多）-------------------
        # 初始化lb和ub数组
        lb = np.full(3*n, -np.inf)
        ub = np.full(3*n, np.inf)
        # 设置规划起点的约束
        k = start_constrain_k # 对起点约束做适当的放宽
        lb[0] = plan_start_l - abs(plan_start_l) * k
        lb[1] = plan_start_dl - abs(plan_start_dl) * k
        lb[2] = plan_start_ddl - abs(plan_start_ddl) * k
        ub[0] = plan_start_l + abs(plan_start_l) * k
        ub[1] = plan_start_dl + abs(plan_start_dl) * k
        ub[2] = plan_start_ddl + abs(plan_start_ddl) * k
        # 考虑到规划起点l可能比较偏，为了到达满足车道边界约束的第二个点，需要的dl和ddl可能较大，这里暂时放开对起点的dl，ddl约束，靠cost限制
        lb[1] = -np.inf
        lb[2] = -np.inf
        ub[1] = np.inf
        ub[2] = np.inf
        
        # 设置其他点的dl, ddl约束
        for i in range(1, n):  
            lb[3*i + 1] = -2
            ub[3*i + 1] = 2
            lb[3*i + 2] = -0.1
            ub[3*i + 2] = 0.1

        # 目标函数生成 0.5*x'Hx + f'*x = min ----------------------------------------------
        # 初始化H矩阵的各个部分
        H_L = np.zeros((3*n, 3*n)) 
        H_DL = np.zeros((3*n, 3*n))
        H_DDL = np.zeros((3*n, 3*n))
        H_DDDL = np.zeros((n-1, 3*n))
        H_CENTRE = np.zeros((3*n, 3*n))
        H_L_END = np.zeros((3*n, 3*n))
        H_DL_END = np.zeros((3*n, 3*n))
        H_DDL_END = np.zeros((3*n, 3*n))
        # 1. H生成
        # 填充H_L, H_DL, H_DDL， H_CENTRE
        for i in range(n):
            H_L[3*i, 3*i] = 1 # 参考线代价
            H_DL[3*i+1, 3*i+1] = 1  # 一阶导平滑代价
            H_DDL[3*i+2, 3*i+2] = 1 # 二阶导平滑代价
        H_CENTRE = np.copy(H_L) # 参考线中心代价
        # 填充H_DDDL
        H_dddl_sub = np.array([0, 0, 1, 0, 0, -1])
        for i in range(n-1):
            row = i
            col = 3*i
            H_DDDL[row, col:col+6] = H_dddl_sub
        # 填充END相关H
        H_L_END[3*n - 3, 3*n - 3] = 1
        H_DL_END[3*n - 2, 3*n - 2] = 1
        H_DDL_END[3*n - 1, 3*n - 1] = 1
        # 生成H ps： dddl项是ddl作差/ds
        H = w_cost_l * np.dot(H_L.T, H_L) + \
            w_cost_dl * np.dot(H_DL.T, H_DL) + \
            w_cost_ddl * np.dot(H_DDL.T, H_DDL) + \
            w_cost_dddl * np.dot(H_DDDL.T, H_DDDL) / ds + \
            w_cost_centre * np.dot(H_CENTRE.T, H_CENTRE) + \
            w_cost_end_l * np.dot(H_L_END.T, H_L_END) + \
            w_cost_end_dl * np.dot(H_DL_END.T, H_DL_END) + \
            w_cost_end_ddl * np.dot(H_DDL_END.T, H_DDL_END)
        H = 2 * H # H需要乘以2，因为二次规划的标准形式中是0.5*x'Hx

        # 2. f生成，考虑末状态约束
        f = np.zeros((3*n, 1))
        centre_line = np.array([(x + y) / 2 for x, y in zip(l_min, l_max)]) # l_min和l_max都是列表，需要转为array,shape=(n,)
        for i in range(n):
            f[3*i] = centre_line[i]
        # 避免中心权重太大
        for i in range(n):
            if(abs(f[i] > 0.3)):
                f[i] *= w_cost_centre
        # 期望末状态约定
        end_l_desire = 0
        end_dl_desire = 0
        end_ddl_desire = 0
        # 设置终点状态的权重
        f[-3] -= 2 * end_l_desire * w_cost_end_l
        f[-2] -= 2 * end_dl_desire * w_cost_end_dl
        f[-1] -= 2 * end_ddl_desire * w_cost_end_ddl


        # 求解二次规划问题 ----------------------------------------------
        x = cp.Variable(3*n)  # 决策变量
        objective = cp.Minimize(0.5 * cp.quad_form(x, H) + f.T @ x) # 目标函数
        constraints = [A_total @ x <= b_total, Aeq @ x == Beq, x >= lb, x <= ub] # 约束条件
        # 定义优化问题
        prob = cp.Problem(objective, constraints)
        # 求解优化问题
        result = prob.solve(solver=cp.OSQP) # 指定求解器
        # 初始化结果列表
        qp_path_l = []
        qp_path_dl = []
        qp_path_ddl = []
        # 检查问题是否成功解决
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            print("PATH_QP Problem solved successfully.")
            for i in range(n):
                qp_path_l.append(x.value[3*i])     # 提取位置
                qp_path_dl.append(x.value[3*i + 1]) # 提取速度
                qp_path_ddl.append(x.value[3*i + 2])# 提取加速度
            qp_path_s = np.arange(n) * ds + plan_start_s
            # 返回路径规划结果
            # print("qp_path_l:", qp_path_l)
            # print("qp_path_dl:", qp_path_dl)
            # print("qp_path_ddl:", qp_path_ddl)
            # print("qp_path_s:", qp_path_s)
        else:
            print("Problem solving failed with status:", prob.status)
            # √ TODO: 给一个规划失败后的备用方案
            # 如果规划失败，路径就保持plan_start
            for i in range(n):
                qp_path_l.append(plan_start_l)
                qp_path_dl.append(0) 
                qp_path_ddl.append(0)
            qp_path_s = np.arange(n) * ds + plan_start_s

        plt.clf()
        plt.plot(qp_path_s, qp_path_l)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        file_name_prefix = "/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/test_fig_LS_QP"
        file_extension = ".png"
        # 拼接完整的文件名
        file_name = f"{file_name_prefix}/LSmap_{formatted_datetime}{file_extension}"
        plt.savefig(file_name)
        plt.close()
            
        return qp_path_s, qp_path_l, qp_path_dl, qp_path_ddl

    def densitification(self, qp_path_s, qp_path_l, qp_path_dl, qp_path_ddl):
        '''
        对优化得到的n个轨迹点通过插值增密到n_final
        :param n 目标轨迹点个数
        '''
        # 初始化
        n_init = len(qp_path_s) 
        n = self._n_final
        qp_path_s_final = np.zeros(n)
        qp_path_l_final = np.zeros(n)
        qp_path_dl_final = np.zeros(n)
        qp_path_ddl_final = np.zeros(n)
        # 计算增密后的ds
        ds = (qp_path_s[-1] - qp_path_s[0]) / (n - 1)
        index = 1
        for i in range(n):
            x = qp_path_s[0] + i * ds
            qp_path_s_final[i] = x
            # 找到x对应的在原始路径中的位置
            while x >= qp_path_s[index] and index < n_init - 1:
                index += 1
            # 计算前一个点和后一个点的索引
            pre = index - 1
            cur = index
            # 计算前一个点和后一个点的 l, l', l'', 以及两点间的ds
            delta_s = x - qp_path_s[pre]
            l_pre = qp_path_l[pre]
            dl_pre = qp_path_dl[pre]
            ddl_pre = qp_path_ddl[pre]
            ddl_cur = qp_path_ddl[cur] if cur < n_init else qp_path_ddl[-1]
            # 使用分段二次插值来增密
            qp_path_l_final[i] = l_pre + dl_pre * delta_s + (1/3) * ddl_pre * delta_s**2 + (1/6) * ddl_cur * delta_s**2
            qp_path_dl_final[i] = dl_pre + 0.5 * ddl_pre * delta_s + 0.5 * ddl_cur * delta_s
            qp_path_ddl_final[i] = ddl_pre + (ddl_cur - ddl_pre) * delta_s / (qp_path_s[cur] - qp_path_s[pre]) if cur < n_init else ddl_pre
            # 准备下一次迭代
            if index > 1:  # 确保index不会变成负数
                index -= 1

        return qp_path_s_final, qp_path_l_final, qp_path_dl_final, qp_path_ddl_final



    def densitification(self, qp_path_s, qp_path_l, qp_path_dl, qp_path_ddl):
        '''
        对优化得到的n个轨迹点通过插值增密到n_final
        :param n 目标轨迹点个数
        '''
        # 初始化
        n_init = len(qp_path_s) 
        n = self._n_final
        qp_path_s_final = np.zeros(n)
        qp_path_l_final = np.zeros(n)
        qp_path_dl_final = np.zeros(n)
        qp_path_ddl_final = np.zeros(n)
        # 计算增密后的ds
        ds = (qp_path_s[-1] - qp_path_s[0]) / (n - 1)
        index = 1
        for i in range(n):
            x = qp_path_s[0] + i * ds
            qp_path_s_final[i] = x
            # 找到x对应的在原始路径中的位置
            while x >= qp_path_s[index] and index < n_init - 1:
                index += 1
            # 计算前一个点和后一个点的索引
            pre = index - 1
            cur = index
            # 计算前一个点和后一个点的 l, l', l'', 以及两点间的ds
            delta_s = x - qp_path_s[pre]
            l_pre = qp_path_l[pre]
            dl_pre = qp_path_dl[pre]
            ddl_pre = qp_path_ddl[pre]
            ddl_cur = qp_path_ddl[cur] if cur < n_init else qp_path_ddl[-1]
            # 使用分段二次插值来增密
            qp_path_l_final[i] = l_pre + dl_pre * delta_s + (1/3) * ddl_pre * delta_s**2 + (1/6) * ddl_cur * delta_s**2
            qp_path_dl_final[i] = dl_pre + 0.5 * ddl_pre * delta_s + 0.5 * ddl_cur * delta_s
            qp_path_ddl_final[i] = ddl_pre + (ddl_cur - ddl_pre) * delta_s / (qp_path_s[cur] - qp_path_s[pre]) if cur < n_init else ddl_pre
            # 准备下一次迭代
            if index > 1:  # 确保index不会变成负数
                index -= 1

        return qp_path_s_final, qp_path_l_final, qp_path_dl_final, qp_path_ddl_final
    
    def planning_on_frenet(self):
        '''planning主函数'''
        # 1 get planning start state
        start_s, start_l, start_s_dot, start_l_dot, \
            start_dl, start_l_dot2, \
            start_s_dot2, start_ddl = self.generate_planning_start_state()
        # 2 generate convec space for dp solving TODO: considering static obstacle or obsatcle with low speed
        l_max, l_min = self.generate_convex_space() 
        # QP solving
        qp_path_s, qp_path_l, qp_path_dl, qp_path_ddl = self.path_planning_QP(start_s, start_l,\
                                                            start_dl, start_ddl, l_min, l_max)
        # 3 densitification
        # qp_path_s_final, qp_path_l_final, \
        #     qp_path_dl_final, qp_path_ddl_final = self.densitification(qp_path_s, qp_path_l, \
        #                                                                qp_path_dl, qp_path_ddl)
        
        return qp_path_l, qp_path_dl, qp_path_ddl, qp_path_s

