import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("PJRT_NPROC", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from shared_types.action import FollowPath
from autonomy.utils.dynamics import INPUT_DIMENSION
from autonomy.utils.reference_trajectory import process_path, get_local_reference
from autonomy.utils.mppi_step import mppi_step
from autonomy.utils.dynamics import dynamics_model
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

# The speed bounds MPPI samples within (lower_bounds / upper_bounds in mppi_step)
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

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        for name, default in CONTROLLER_PARAMS.items():
            setattr(self, name, self.declare_parameter(name, default).value)
        problem = controller_param_problem(**{name: getattr(self, name) for name in CONTROLLER_PARAMS})
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

    def on_set_parameters(self, params):
        settings = {name: getattr(self, name) for name in CONTROLLER_PARAMS}
        settings.update((p.name, p.value) for p in params if p.name in settings)
        problem = controller_param_problem(**settings)
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
    
        U_opt = mppi_step(
            key=iter_key,
            x0=future_state,
            U_nominal=self.U_nominal,
            ref_traj=ref_traj_jax,
            v_target=V_TARGET,
        )
    
        # [min_v, min_delta]
        lower_bounds = jnp.array([self.min_speed, -0.6])
        upper_bounds = jnp.array([self.max_speed, 0.6])
        U_opt = jnp.clip(U_opt, lower_bounds, upper_bounds)
    
        u = U_opt[0]
        self.publish_command(u)
        self.last_u = u
    
        # Warm start the next solve by shifting the horizon forward one step
        self.U_nominal = jnp.roll(U_opt, shift=-1, axis=0)
        self.U_nominal = self.U_nominal.at[-1].set(U_opt[-1])

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
