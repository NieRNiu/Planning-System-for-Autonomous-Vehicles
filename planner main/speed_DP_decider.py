from datetime import datetime
import numpy as np
import math
from typing import List, Type, Optional, Tuple
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.planner.project2.frame_transform import get_match_point, cal_project_point
from shapely.geometry import Point, LineString
from scipy.interpolate import interp1d

class SPEED_DP_DECIDER_v1:
    '''
    课程给出的速度DP算法,稍作修改
    '''
    def __init__(self, 
                 ego_state: EgoState,
                 objects: List[TrackedObject],
                 path_idx2s: List[float], 
                 path_x: List[float], 
                 path_y: List[float], 
                 path_heading: List[float], 
                 path_kappa: List[float],
                 total_time: float, 
                 step: float,
                 max_v:float,
                 ego_v: float,
                 max_acc: float,
                 max_dec: float,
                 delta_s=0.5) -> None:
        # 处理objects信息为prject2参考dp代码所需格式：
        self._obs_trajectory: List[List[List[float]]] = [] # 最内层list为轨迹点，中间层list是轨迹点列表组成轨迹，外层list是多个obj
        self._obs_radius: List[float] = [] # 每个智能体的障碍物避碰半径
        for object in objects:
            single_traj: List[List[float]] = []# 轨迹点列表
            # TODO: 针对多模态预测的情况修改框架
            for waypoint in object.predictions[0].waypoints:
                traj_point: List[float] = [0, 0, 0] # traj[0]=t, traj[1]=x, traj[2]=y
                traj_point[0] = waypoint.time_point.time_s - ego_state.time_point.time_s
                traj_point[1] = waypoint.center.x
                traj_point[2] = waypoint.center.y
                single_traj.append(traj_point)
            self._obs_trajectory.append(single_traj)
            radius = np.linalg.norm(np.array([object.box.half_length, object.box.half_width]))
            self._obs_radius.append(radius)
        # self._obs_trajectory = obs_trajectory
        # self._obs_radius = obs_radius
        self._path_idx2s = path_idx2s
        self._path_x = path_x
        self._path_y = path_y
        self._path_heading = path_heading
        self._path_kappa = path_kappa
        self._total_t = total_time
        self._delta_t = step
        # self._ego_half_width = ego_half_width
        # self._ego_length = ego_length
        self._ego_half_width = ego_state.car_footprint.half_width
        self._ego_length = ego_state.car_footprint.length
        self._max_v = max_v
        self._max_acc = max_acc
        self._max_dec = max_dec
        self._w_cost_ref_speed = 40
        self._reference_speed = max_v
        self._w_cost_accel = 10
        self._w_cost_obs = 200
        self._plan_start_s_dot = ego_v
        self._obs_ts: List[LineString] = [] # List[[(x1, y1), (x2, y2), (x3, y3), ...],.....]
        self._obs_interp_ts: List[interp1d] = []
        self._delta_s = delta_s


    def dynamic_programming(self) ->Tuple[float]:
        # 根据障碍物轨迹和路径规划结果，将障碍物轨迹映射到ST图
        self._obs_ts = []

        import matplotlib.pyplot as plt
        # 绘制曲线
        plt.clf()

        for trajectory in self._obs_trajectory:
            # 每个agent的trajectory
            points_ts = []
            for state in trajectory:
                t = state[0]
                x = state[1]
                y = state[2]
                # 计算每个点在path上的s和l
                match_point_index_set = get_match_point([x], [y], self._path_x, self._path_y)
                proj_x_set, proj_y_set, proj_heading_set, _, proj_s_set = cal_project_point(\
                                    [x], [y], match_point_index_set, self._path_x, self._path_y, self._path_heading, self._path_kappa, self._path_idx2s)
                s = proj_s_set[0]
                n_r = np.array([-math.sin(proj_heading_set[0]), math.cos(proj_heading_set[0])]) # 投影点法向量
                r_h = np.array([x, y])  
                r_r = np.array([proj_x_set[0], proj_y_set[0]]) # 投影点向量
                l = np.dot((r_h - r_r), n_r)
                if s >= 0 and s <= self._path_idx2s[-1] and abs(l) <= self._ego_half_width:
                    points_ts.append((t, s))
                # if s <= self._path_idx2s[-1]: # and abs(l) <= self._ego_half_width:
                #     points_ts.append((t, s))
            if len(points_ts) > 1:
                line_ts = LineString(points_ts)
                interp_ts = interp1d([ts[0] for ts in points_ts], [ts[1] for ts in points_ts])
                self._obs_ts.append(line_ts)
                self._obs_interp_ts.append(interp_ts)

            t_points = [point[0] for point in points_ts]
            s_points = [point[1] for point in points_ts]
            plt.plot(t_points, s_points, color='red')  # 连接成连续曲线并绘制在曲线图上，使用红色线条
        # 获取当前绝对时间，并将其格式化为字符串
        # plt.grid(True)
        # current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # # 构造文件名，包括路径、文件名和后缀
        # file_name = f'/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/plot_test/ST_DP_RES_{current_time_str}.png'
        # plt.savefig(file_name)
        # # plt.close()

        
        # print('ST图投影完成')
        # S T撒点，T撒点要和后续的速度规划保持一致，S的最大值也和后续的速度规划保持一致(max_v * total_time)
        t_list = np.arange(self._delta_t, self._total_t, self._delta_t) # t = 0不必搜索
        t_list = np.append(t_list, self._total_t)
        # max_s = self._max_v * self._total_t # 13m限速，就要搜13*8=104m，有点大了
        # delta_s = self._delta_s
        # s_list = np.arange(0, max_s, delta_s)
        # s_list = np.append(s_list, max_s)
        # 稀疏采样可以加快速度，但也容易导致找不到符合约束的dp_s
        # third = int(max_s / 3) 
        # s_list1 = np.arange(0, third, delta_s)
        # s_list2 = np.arange(third, max_s, 4 * delta_s)
        # s_list = np.concatenate((s_list1, s_list2))
        # s_list = np.append(s_list, max_s)
        # 改一下稀疏策略1 s.size = 50
        # max_s = 55
        # s_list_1 = np.arange(0, 5, 0.5) # [0~4.5]m # n=10
        # s_list_2 = np.arange(5, 15, 1) # [5~14]m # n=10
        # s_list_3 = np.arange(15, 30, 1.5) # [15~28.5]m # n=10
        # s_list_4 = np.arange(30, 55, 2.5) #[30~52.5]m # n=10
        # # s_list_5 = np.arange(55, 90, 4.5) #[55~86.5]m # n=10
        # s_list = np.concatenate((s_list_1, s_list_2, s_list_3, s_list_4))
        # s_list = np.append(s_list, max_s)
        # 稀疏策略2 s.size = 30
        max_s = 55
        s_list_1 = np.arange(0, 5, 0.5) # [0~4.5]m # n=10
        s_list_2 = np.arange(5, 15, 1) # [5~14]m # n=10
        s_list_3 = np.arange(15, 55, 4) # [15~51]m # n=10
        # s_list_4 = np.arange(30, 5, 2.5) #[30~52.5]m # n=10
        # s_list_5 = np.arange(55, 90, 4.5) #[55~86.5]m # n=10
        s_list = np.concatenate((s_list_1, s_list_2, s_list_3))
        s_list = np.append(s_list, max_s)
        # 稀疏策略3 s.size = 22
        # max_s = 55 
        # s_list_1 = np.arange(0, 10, 1) # [0~9]m # n=10
        # s_list_2 = np.arange(10, 20, 2) # [10~18]m # n=5
        # s_list_3 = np.arange(20, 56, 6) # [20~50]m # n=6
        # # s_list_4 = np.arange(30, 5, 2.5) #[30~52.5]m # n=10
        # # s_list_5 = np.arange(55, 90, 4.5) #[55~86.5]m # n=10
        # s_list = np.concatenate((s_list_1, s_list_2, s_list_3))
        # s_list = np.append(s_list, max_s)

        # 保存dp过程的数据
        dp_st_cost =  [[math.inf] * len(t_list) for _ in range(len(s_list))] # [[t]]
        dp_st_s_dot =  [[0] * len(t_list) for _ in range(len(s_list))] # [[t]] 表示从起点开始到(i,j)点的最优路径的末速度
        dp_st_node =  [[0] * len(t_list) for _ in range(len(s_list))] # 记录

        # 计算从dp起点到第一列的cost
        print("start DP search")
        for i in range(len(s_list)):
            dp_st_cost[i][0] = self._CalcDpCost(-1, -1, i, 0, s_list, t_list,dp_st_s_dot)
            # 计算第一列所有节点的的s_dot，并存储到dp_st_s_dot中 第一列的前一个节点只有起点
            dp_st_s_dot[i][0] = (s_list[i] - self._ego_length/2) / t_list[0]

        # 动态规划主程序
        for i in np.arange(1, len(t_list)):
            # i 为列循环
            for j in range(len(s_list)):
                # j 为行循环
                # 当前行为 j 列为 i
                cur_row = j
                cur_col = i
                # 遍历前一列
                for k in range(len(s_list)):
                    pre_row = k
                    pre_col = i - 1
                    # 计算边的代价 其中起点为pre_row,pre_col 终点为cur_row cur_col
                    cost_temp = self._CalcDpCost(pre_row, pre_col, cur_row, cur_col, s_list, t_list, dp_st_s_dot)
                    if cost_temp + dp_st_cost[pre_row][pre_col] < dp_st_cost[cur_row][cur_col]:
                        dp_st_cost[cur_row][cur_col] = cost_temp + dp_st_cost[pre_row][pre_col]
                        # 计算最优的s_dot
                        s_start, t_start = self._CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = self._CalcSTCoordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row][cur_col] = (s_end - s_start) / (t_end - t_start)
                        # 将最短路径的前一个节点的行号记录下来
                        dp_st_node[cur_row][cur_col] = pre_row
        
        # 输出dp结果
        # 输出初始化
        dp_speed_s = [-1] * len(t_list)
        dp_speed_t = [-1] * len(t_list)
        # 找到dp_node_cost 上边界和右边界代价最小的节点
        # 1.找右边界
        min_cost_1 = math.inf
        min_row_1 = math.inf
        for i in range(len(s_list)):
            # 遍历右边界，找到最小点(min_row_1, len(t_list) - 1)
            if dp_st_cost[i][-1] <= min_cost_1:
                min_cost_1 = dp_st_cost[i][-1]
                min_row_1 = i
        # 2.找上边界
        min_cost_2 = math.inf
        min_col_2 = math.inf
        for j in range(len(t_list) - 1):
            # 遍历上边界，找到最小点(min_col_2, len(s_list) - 1)
            if dp_st_cost[- 1][j] <= min_cost_2:
                min_cost_2 = dp_st_cost[-1][j]
                min_col_2 = j
        # 3. 比较
        if(min_cost_1 <= min_cost_2):
            min_row = min_row_1
            min_col = len(t_list) - 1
        else:
            min_row = len(s_list) - 1
            min_col = min_col_2
        # 先把终点的ST输出出来
        s, t = self._CalcSTCoordinate(min_row, min_col, s_list, t_list)
        dp_speed_s[min_col] = s
        dp_speed_t[min_col] = t
        # 反向回溯
        while min_col != 0:
            pre_row = dp_st_node[min_row][min_col]
            pre_col = min_col - 1
            s, t = self._CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
            dp_speed_s[pre_col] = s
            dp_speed_t[pre_col] = t
            min_row = pre_row
            min_col = pre_col
        
        s_lb = [0] * len(t_list)
        s_ub = [max_s] * len(t_list)
        # 根据dp结果和_obs_ts，计算s_lb, s_ub
        for i in range(len(self._obs_ts)):
            obs_ts = self._obs_ts[i]
            obs_interp_ts = self._obs_interp_ts[i]
            bounds = obs_ts.bounds
            obs_t_min = bounds[0]
            obs_t_max = bounds[2]
            for j in range(len(t_list)):
                t = t_list[j]
                if t < obs_t_min or t > obs_t_max:
                    continue
                s = dp_speed_s[j] if dp_speed_s[j] != -1 else max_s # 没到horizen_time，后面的点都按最大值处理
                # 计算obs在当前t的s，判断决策，输出边界
                obs_s = obs_interp_ts(t)
                if s < obs_s and s != -1: # 避让改上界，dp不一定到horizentime
                    s_ub[j] = min(s_ub[j], max(0, obs_s - self._ego_length - self._obs_radius[i]))
                    if s_ub[j] < s_lb[j] + 0.5:
                        s_ub[j] = s_lb[j] + 0.5
                    # s_ub[j] = min(s_ub[j], max(0, obs_s))
                else: # 超车改下界
                    s_lb[j] = max(s_lb[j], min(max_s, obs_s + self._ego_length + self._obs_radius[i]))
                    # s_lb[j] = max(s_lb[j], min(max_s, obs_s))
                    if s_lb[j] > s_ub[j] - 0.5:
                        s_lb[j] = s_ub[j] - 0.5

        # for i in range(1, len(dp_speed_s) - 1):
        #     if(dp_speed_s[i] == 0):
        #         print('error!')

        # 也许没dp满8s，过滤一下
        filtered_dp_speed_t = [t for t, s in zip(dp_speed_t, dp_speed_s) if s != -1]
        filtered_dp_speed_s = [s for s in dp_speed_s if s != -1]
        plt.plot(filtered_dp_speed_t, filtered_dp_speed_s, color='blue')
        # plt.plot(dp_speed_t, dp_speed_s, color='blue')

        # 添加标题和标签
        plt.title('S vs Time')
        plt.xlabel('T')
        plt.ylabel('S')
        plt.ylim((-5, max_s))

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.show()

        # # 获取当前绝对时间，并将其格式化为字符串
        # current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # # 构造文件名，包括路径、文件名和后缀
        # file_name = f'/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/plot_test/ST_DP_RES_{current_time_str}.png'
        # plt.savefig(file_name)
        # plt.close()
        
        import os
        # from datetime import datetime
        # 创建一个文件夹用于保存图片
        # if not os.path.exists("images"):
        #     os.mkdir("images")
        # 获取当前时间并格式化
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

        # 设置文件名前缀和扩展名
        file_name_prefix = "/home/dx/ZJU_research/project/nuplan/nuplan-devkit/nuplan/planning/simulation/planner/project2/test_fig_ST_DP"
        file_extension = ".png"

        # 拼接完整的文件名
        file_name = f"{file_name_prefix}/STmap_{formatted_datetime}{file_extension}"

        plt.savefig(file_name)
        # plt.close() # 不close，接着画QP
        print('s_lb: ', s_lb)
        print('s_ub: ', s_ub)

        # dp_s_out = dp_speed_s
        dp_s_out = [s - self._ego_length/2 for s in dp_speed_s]
        dp_t_out = t_list
        print("speed DP finished")
        return s_lb, s_ub, dp_s_out, dp_t_out, dp_st_s_dot


    def _CalcDpCost(self, row_start, col_start, row_end, col_end, s_list, t_list, dp_st_s_dot):
        """
        该函数将计算链接两个节点之间边的代价
        :param 边的起点的行列号row_start,col_start 边的终点行列号row_end,col_end
        :param s_list,t_list
        :dp_st_s_dot 用于计算加速度
        :return 边的代价
        """
        # 首先计算终点的st坐标
        s_end, t_end = self._CalcSTCoordinate(row_end, col_end, s_list, t_list)
        # 规定起点的行列号为-1 
        if row_start == -1:
            # 边的起点为dp的起点
            s_start = self._ego_length/2
            # s_start = 0
            t_start = 0
            s_dot_start = self._plan_start_s_dot
        else:
            # 边的起点不是dp的起点
            s_start, t_start = self._CalcSTCoordinate(row_start, col_start, s_list, t_list)
            s_dot_start = dp_st_s_dot[row_start][col_start]
        cur_s_dot = (s_end - s_start) / (t_end - t_start)
        cur_s_dot2 = (cur_s_dot - s_dot_start)/(t_end - t_start)
        # 计算推荐速度代价
        cost_ref_speed = self._w_cost_ref_speed * (cur_s_dot - self._reference_speed) ** 2
        # if cur_s_dot <= self._reference_speed and cur_s_dot >= 0:
        #     cost_ref_speed = self._w_cost_ref_speed * (cur_s_dot - self._reference_speed)**2
        # elif cur_s_dot > self._reference_speed:
        #     cost_ref_speed = 100 * self._w_cost_ref_speed * (cur_s_dot - self._reference_speed)**2 # 尽量让dp有解
        # else:
        #     cost_ref_speed = math.inf
        # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
        # cost_accel = self._w_cost_accel * cur_s_dot2 ** 2
        if cur_s_dot2 <= self._max_acc and cur_s_dot2 >= self._max_dec:
            cost_accel = self._w_cost_accel * cur_s_dot2**2
        else:
            # 超过车辆动力学限制，代价会增大很多倍
            cost_accel = 1000 * self._w_cost_accel * cur_s_dot2**2
            # cost_accel = math.inf
        cost_obs = self._CalcObsCost(s_start,t_start,s_end,t_end)
        cost = cost_obs + cost_accel + cost_ref_speed

        if(cost == math.inf):
            print('error: cost=inf!!')
        return cost

    def _CalcObsCost(self, s_start, t_start, s_end, t_end):
        """
        该函数将计算边的障碍物代价
        :param 边的起点终点s_start,t_start,s_end,t_end
        :return 边的障碍物代价obs_cost
        """
        # 输出初始化
        obs_cost = 0
        # 边的采样点的个数
        n = 5
        # 采样时间间隔
        dt = (t_end - t_start)/(n - 1)
        # 边的斜率
        k = (s_end - s_start)/(t_end - t_start)
        for i in range(n+1):
            # 计算采样点的坐标
            t = t_start + i * dt
            s = s_start + k * i * dt
            point_ts = Point(t, s)
            # 遍历所有障碍物
            for j in range(len(self._obs_ts)):
                obs_ts = self._obs_ts[j]
                bounds = obs_ts.bounds
                obs_t_min = bounds[0]
                obs_t_max = bounds[2]
                if t < obs_t_min or t > obs_t_max:
                    continue
                obs_interp_ts = self._obs_interp_ts[j]
                obs_s = obs_interp_ts(t)
                dis = s - obs_s
                obs_cost = obs_cost + self._CalcCollisionCost(self._w_cost_obs, dis, j)
                # # 计算点到st折线的最短距离
                # min_dis = point_ts.distance(self._obs_ts[j])
                # obs_cost = obs_cost + self._CalcCollisionCost(self._w_cost_obs, min_dis, j)
        return obs_cost
    
    def _CalcCollisionCost(self, w_cost_obs, min_dis, obs_idx):
        collision_cost = 0
        obs_radius = self._obs_radius[obs_idx]
        buffer = obs_radius + self._ego_length/2
        if abs(min_dis) < buffer:
            collision_cost = w_cost_obs * 10000
            # collision_cost = math.inf # 用inf在稀疏采样时候容易崩溃
        elif abs(min_dis) > buffer and abs(min_dis) < buffer*3:
            collision_cost = w_cost_obs**((buffer - abs(min_dis))/buffer + 2)
        else:
            collision_cost = 0
        return collision_cost
    
    def _CalcSTCoordinate(self, row, col, s_list, t_list):
        s = s_list[row]
        t = t_list[col]
        return s, t







        
        