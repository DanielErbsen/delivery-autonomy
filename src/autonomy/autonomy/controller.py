import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("PJRT_NPROC", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from shared_types.action import FollowPath
from autonomy.utils.dynamics import INPUT_DIMENSION
from autonomy.utils.reference_trajectory import process_path, get_local_reference
from autonomy.utils.mppi_step import charged_steps, crop_window, free_window, mppi_step
from autonomy.utils.dynamics import dynamics_model
from autonomy.utils.costmap_speed import (
    GOVERNOR_PARAMS, Costmap, end_taper_speed, governor_param_problem, path_ahead, path_progress,
    speed_limit)
import time
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.task import Future
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import jax
import jax.numpy as jnp
import numpy as np

CONTROLLER_PARAMS = {
    'max_speed': 1.5,
    'min_speed': -0.5,
    'xy_goal_tolerance': 0.25,
    'yaw_goal_tolerance': 0.15,
}

# The speeds MPPI is tuned for. Up to 1.5 m/s its rollouts stay inside the costmap window.
MPPI_MIN_SPEED, MPPI_MAX_SPEED = -0.5, 1.5

def controller_param_problem(max_speed, min_speed, xy_goal_tolerance, yaw_goal_tolerance):
    """Return why these settings are unusable, or None when they are fine."""
    if not 0.0 < max_speed <= MPPI_MAX_SPEED:
        return f'max_speed must be in (0, {MPPI_MAX_SPEED}] m/s, got {max_speed}'
    if not MPPI_MIN_SPEED <= min_speed <= 0.0:
        return f'min_speed must be in [{MPPI_MIN_SPEED}, 0] m/s, got {min_speed}'
    if not xy_goal_tolerance > 0.0:
        return f'xy_goal_tolerance must be positive, got {xy_goal_tolerance}'
    if not yaw_goal_tolerance > 0.0:
        return f'yaw_goal_tolerance must be positive, got {yaw_goal_tolerance}'
    return None

def param_problem(settings):
    """Check a full set of controller and speed-limit settings (a dict by parameter name)."""
    return (controller_param_problem(**{name: settings[name] for name in CONTROLLER_PARAMS})
            or governor_param_problem(**{name: settings[name] for name in GOVERNOR_PARAMS}))

class ControllerActionServer(Node):
    def __init__(self):
        super().__init__('controller_server')

        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.state_cb_group = MutuallyExclusiveCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowPath,
            '/follow_path',
            self.execute_callback,
            callback_group=self.action_cb_group,
        )


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter("state_topic", "/state_estimation/odom")
        state_topic = self.get_parameter("state_topic").value
        
        self.current_state = None
        self.state_subscription = self.create_subscription(
            Odometry,
            state_topic,
            self.process_state_callback,
            10,
            callback_group=self.state_cb_group,
        )

        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        costmap_topic = self.get_parameter("costmap_topic").value
        
        self.costmap = None
        self.costmap_arrival = None
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self.process_costmap_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self.state_cb_group,
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        self.param_names = (*CONTROLLER_PARAMS, *GOVERNOR_PARAMS)
        for name, default in {**CONTROLLER_PARAMS, **GOVERNOR_PARAMS}.items():
            setattr(self, name, self.declare_parameter(name, default).value)
        problem = param_problem({name: getattr(self, name) for name in self.param_names})
        if problem:
            raise ValueError(problem)
        self.add_on_set_parameters_callback(self.on_set_parameters)

        self.T = 50 # TODO: make param
        self.steering_wheel_base = 0.5  # Must match <wheel_base> in robot.urdf.xacro
        self.dt = 0.1
        self.rng = jax.random.PRNGKey(42)
        self.U_nominal = jnp.zeros((self.T, INPUT_DIMENSION))
        
        self.active_goal = False
        self.current_path = None
        self.control_timer = self.create_timer(self.dt, self.control_loop_callback)
        
        self.progress_idx = 0
        self.last_u = None
        self._goal_done = None
        self._map_to_odom = None
        self._odom_frame = None
        self._speed_status = None
        self._free_window = free_window()

        self.get_logger().info('Controller Action Server initialized.')

    def process_state_callback(self, odom_msg: Odometry):
        try:
            t = self.tf_buffer.lookup_transform('map', odom_msg.header.frame_id, rclpy.time.Time())
            q = t.transform.rotation
            self._map_to_odom = (
                t.transform.translation.x,
                t.transform.translation.y,
                np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2)),
            )
        except TransformException as ex:
            if self._map_to_odom is None:
                self.get_logger().warning(
                    f'No map->{odom_msg.header.frame_id} transform yet: {ex}', throttle_duration_sec=2.0)
                return None
        map_x, map_y, map_yaw = self._map_to_odom
        self._odom_frame = odom_msg.header.frame_id
    
        # Extract Position and Yaw (theta) in Map frame
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        c, s = np.cos(map_yaw), np.sin(map_yaw)
        x = map_x + c * p.x - s * p.y
        y = map_y + s * p.x + c * p.y
        theta = map_yaw + np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
    
        v = odom_msg.twist.twist.linear.x
        omega = odom_msg.twist.twist.angular.z
    
        # TODO: get this somehow
        b_omega = 0
    
        self.current_state = np.array([x, y, theta, v, omega, b_omega])

    def process_costmap_callback(self, msg: OccupancyGrid):
        self.costmap = Costmap.from_msg(msg)
        self.costmap_arrival = self.get_clock().now()

    def on_set_parameters(self, params):
        settings = {name: getattr(self, name) for name in self.param_names}
        settings.update((p.name, p.value) for p in params if p.name in settings)
        problem = param_problem(settings)
        if problem:
            return SetParametersResult(successful=False, reason=problem)
        for name, value in settings.items():
            setattr(self, name, value)
        return SetParametersResult(successful=True)

    def publish_command(self, u):
        v_cmd, delta = u
    
        v_cmd_float = float(v_cmd)
        delta_float = float(delta)
    
        max_steer = 0.6
        v_cmd_float = max(min(v_cmd_float, self.max_speed), self.min_speed)
        delta_clipped = np.clip(delta_float, -max_steer, max_steer)
        omega = (v_cmd_float / self.steering_wheel_base) * np.tan(delta_clipped)
    
        cmd_vel = Twist()
    
        cmd_vel.linear.x = v_cmd_float
        cmd_vel.angular.z = omega
    
        self.cmd_pub.publish(cmd_vel)

    async def execute_callback(self, goal_handle):
        path = process_path(goal_handle.request.path)
    
        # Only the newest goal stays open: the one it replaces finishes now
        self._finish_goal(reached=False)
    
        if len(path[0]) == 0:
            self.get_logger().info('Received an empty path. Stopping robot.')
            self.active_goal = False
            self.publish_command(np.zeros(INPUT_DIMENSION))
            goal_handle.succeed()
            return FollowPath.Result(success=True)
    
        self.get_logger().info('Received a new path to follow!')
        self.current_path = path
        self.progress_idx = 0
        done = self._goal_done = Future()
        self.active_goal = True
    
        # Wait without blocking the single-threaded executor
        reached = await done
    
        if reached:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return FollowPath.Result(success=reached)
    
    def _finish_goal(self, reached):
        done, self._goal_done = self._goal_done, None
        if done is not None and not done.done():
            done.set_result(reached)

    def control_loop_callback(self):
        if not self.active_goal or self.current_state is None:
            return
    
        path_xy, path_theta_unwrapped, path_s = self.current_path
    
        # doesn't have to be calculated every step but minimal overhead
        goal_x, goal_y = path_xy[-1]
        goal_theta = path_theta_unwrapped[-1]
    
        V_TARGET = self.max_speed
    
        # Assuming state is [x, y, theta, v, omega, b_omega]
        curr_x, curr_y = self.current_state[0], self.current_state[1]
        curr_theta = self.current_state[2]
        
        # Calculate errors
        dist_to_goal = np.hypot(goal_x - curr_x, goal_y - curr_y)
        yaw_err = abs((curr_theta - goal_theta + np.pi) % (2 * np.pi) - np.pi)
        
        # Check termination condition
        if dist_to_goal < self.xy_goal_tolerance and yaw_err < self.yaw_goal_tolerance:
            # Stop the motors
            self.publish_command(np.zeros(INPUT_DIMENSION))
        
            self.active_goal = False
            self._finish_goal(reached=True)
            return
    
        # Optimise from where the robot will be once this solve finishes, not where
        # it was when the solve started
        future_state = dynamics_model(self.current_state, self.last_u, dt=self.dt) if self.last_u is not None else self.current_state
    
        # Brake in time to stop at the end of the path (see End of Path)
        remaining = path_s[-1] - path_progress(future_state, path_xy, path_s, self.progress_idx)
        V_TARGET = min(V_TARGET, end_taper_speed(remaining, self.goal_decel, self.goal_creep_speed))
    
        # Obstacles near the path ahead lower the speed (see Obstacle Speed Limit).
        # A plain float: a numpy scalar would make jax compile mppi_step a second time.
        V_TARGET = float(min(V_TARGET, self.costmap_speed_limit(future_state, path_xy, path_s)))
        if V_TARGET <= 0.0:
            # Too close: hold still, keep the goal, and look again next tick
            u = np.zeros(INPUT_DIMENSION)
            self.publish_command(u)
            self.last_u = u
            return
    
        ref_traj_np, self.progress_idx = get_local_reference(
            future_state,
            path_xy,
            path_theta_unwrapped,
            path_s,
            self.T,
            self.dt,
            v_target=V_TARGET,
            progress_idx=self.progress_idx,
        )
        ref_traj_jax = jnp.array(ref_traj_np)
    
        self.rng, iter_key = jax.random.split(self.rng)
    
        # [min_v, min_delta]. The speed cap is this tick's target, so a lowered target
        # caps the command too. MPPI samples within the same bounds (see MPPI).
        lower_bounds = jnp.array([self.min_speed, -0.6])
        upper_bounds = jnp.array([V_TARGET, 0.6])
    
        U_opt = mppi_step(
            key=iter_key,
            x0=future_state,
            # The warm start may still hold speeds above a target that just dropped
            U_nominal=jnp.clip(self.U_nominal, lower_bounds, upper_bounds),
            ref_traj=ref_traj_jax,
            v_target=V_TARGET,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            # The costmap round the robot, for MPPI's obstacle cost, which stops counting where
            # a rollout could have reached the end of the path (see Obstacle Cost)
            window=self.mppi_costmap_window(future_state),
            obstacle_steps=charged_steps(remaining, V_TARGET, self.T),
        )
    
        U_opt = jnp.clip(U_opt, lower_bounds, upper_bounds)
    
        u = U_opt[0]
        self.publish_command(u)
        self.last_u = u
    
        # Warm start the next solve by shifting the horizon forward one step
        self.U_nominal = jnp.roll(U_opt, shift=-1, axis=0)
        self.U_nominal = self.U_nominal.at[-1].set(U_opt[-1])

    def costmap_speed_limit(self, state, path_xy, path_s):
        """The fastest the costmap lets the robot go on from `state` along the path."""
        if not self.use_costmap:
            return self.max_speed
        age = costmap_pose = None
        if self.costmap is not None:
            age = (self.get_clock().now() - self.costmap_arrival).nanoseconds * 1e-9
            costmap_pose = {'map': (0.0, 0.0, 0.0),
                            self._odom_frame: self._map_to_odom}.get(self.costmap.frame)
        points = path_ahead(state, path_xy, path_s, self.progress_idx, self.clearance_lookahead)
        limit, clearance, status = speed_limit(
            self.costmap, age, points, costmap_pose, self.max_speed, self.slow_distance,
            self.stop_distance, self.obstacle_creep_speed, self.costmap_timeout, self.crawl_speed)
    
        if status in ('missing', 'unplaced', 'stale'):
            problem = {
                'missing': f'no costmap on {self.costmap_subscription.topic_name} yet',
                'unplaced': f'costmap in frame "{self.costmap.frame if self.costmap else ""}", '
                            f'neither map nor {self._odom_frame}',
                'stale': f'costmap {age:.1f} s old',
            }[status]
            self.get_logger().warning(f'{problem}: speed capped at {limit:.2f} m/s',
                                      throttle_duration_sec=5.0)
        elif status == 'stop' and self._speed_status != 'stop':
            self.get_logger().info(
                f'Obstacle {clearance:.2f} m away near the path ahead: stopped until it clears')
        elif status != 'stop' and self._speed_status == 'stop':
            self.get_logger().info('Path ahead clear: driving on')
        self._speed_status = status
        return limit

    def mppi_costmap_window(self, state):
        """The costmap round `state` for MPPI's obstacle cost, or an all-free window when the
        costmap is off, missing, stale or in a frame the controller cannot place."""
        if not self.use_costmap or self.costmap is None:
            return self._free_window
        age = (self.get_clock().now() - self.costmap_arrival).nanoseconds * 1e-9
        frame_pose = {'map': (0.0, 0.0, 0.0), self._odom_frame: self._map_to_odom}.get(self.costmap.frame)
        if age > self.costmap_timeout or frame_pose is None:
            return self._free_window
        return crop_window(self.costmap, state, frame_pose)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerActionServer()

    # Single-threaded on purpose: a MultiThreadedExecutor under ~100 Hz odometry plus /tf
    # (and the sim's /clock) starves the 10 Hz control timer on the GIL. The action callback
    # is async, so it does not need a thread of its own.
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
