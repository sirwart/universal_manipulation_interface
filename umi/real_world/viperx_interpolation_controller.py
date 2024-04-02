import enum
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class ViperXInterface:
    def __init__(self):
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
        from interbotix_xs_msgs.msg import JointSingleCommand

        self.bot = InterbotixManipulatorXS(robot_model='vx300s', group_name='arm', gripper_name='gripper')
        self._gripper_cmd = JointSingleCommand(name="gripper")

        self.bot.core.robot_reboot_motors("single", "gripper", True)
        self.bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")

        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Velocity', 100)
        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Acceleration', 0)

    def stop(self):
        self.bot.core.robot_set_operating_modes("single", "gripper", "pwm")

        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Velocity', 2000)
        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Acceleration', 300)

    def _pose_matrix_to_cartesian(self, pose_matrix):
        cartesian_coords = pose_matrix[:3, 3]
        rotation = Rotation.from_matrix(pose_matrix[:3, :3])
        rotation_coords = rotation.as_euler('xyz')
        return np.concatenate([cartesian_coords, rotation_coords])

    def get_ee_pose(self):
        return self._pose_matrix_to_cartesian(self.bot.arm.get_ee_pose())

    def update_desired_ee_pose(self, pose):
        current_joint_commands = self.bot.arm.get_joint_commands()
        self.bot.arm.set_ee_pose_components(*pose, custom_guess=current_joint_commands, blocking=False)

    def get_obs(self):
        joint_states = self.bot.core.joint_states
        joint_positions = np.array(joint_states.position[:6])
        joint_velocities = np.array(joint_states.velocity[:6])

        return {
            'ActualTCPPose': self.get_ee_pose(),
            'ActualQ': joint_positions,
            'ActualQd': joint_velocities,
            'TargetTCPPose': self._pose_matrix_to_cartesian(self.bot.arm.get_ee_pose_command())
        }

class ViperXGripperController:
    def __init__(self, controller):
        self.controller = controller

    def start(self, wait=True):
        pass

    def stop(self, wait=True):
        pass

    def start_wait(self):
        pass

    def stop_wait(self):
        pass

    @property
    def is_ready(self):
        return self.controller.is_ready

    def get_state(self):
        return {
            'gripper_position': 1.0,
            'gripper_timestamp': time.time(),
        }

    def get_all_state(self):
        return {
            'gripper_position': np.array([1.0]),
            'gripper_timestamp': np.array([time.time()]),# - self.receive_latency
        }

    def schedule_waypoint(self, pos, target_time):
        pass


class ViperXInterpolationController(mp.Process):

    def __init__(self,
        shm_manager,
        frequency=125,
        launch_timeout=3,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        super().__init__(name='ViperXInterpolationController')

        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.receive_latency = receive_latency

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            'ActualTCPPose',
            'ActualQ',
            'ActualQd',
            'TargetTCPPose',
        ]
        example = dict()
        for key in receive_keys:
            example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ViperXInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start viperX interface
        robot = ViperXInterface()

        try:
            if self.verbose:
                print("[ViperXInterpolationController] Connect to robot")
            
            # init pose
            '''
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )
            '''

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            print(f'{curr_pose=}')

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            '''
            # start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )
            '''

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now)
                #flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)
                flange_pose = tip_pose # XXX

                # send command to robot
                robot.update_desired_ee_pose(flange_pose)

                # update robot state
                state = robot.get_obs()
                   
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        print('received STOP command')
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[ViperXInterpolationController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    print('setting ready_event')
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[ViperXInterpolationController] Actual frequency {1/(time.monotonic() - t_now)}")
        except Exception as e:
            print('viperx exception:', e)
        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.stop()
            self.ready_event.set()

            if self.verbose:
                print(f"[ViperXInterpolationController] Disconnected from robot: {self.robot_ip}")
