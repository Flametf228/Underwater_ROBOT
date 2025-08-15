import cv2
import time
import asyncio
import numpy as np
import pymurapi as pm
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
import logging

logging.basicConfig(level=logging.INFO)

auv = pm.mur_init()

class Color(Enum):
    RED = 1
    ORANGE = 2
    YELLOW = 3
    CHARTREUSE_GREEN = 4
    GREEN = 5
    SPRING_GREEN = 6
    CYAN = 7
    AZURE = 8
    BLUE = 9
    VIOLET = 10
    MAGENTA = 11
    ROSE = 12

class MotorNums(IntEnum):
    LEFT_FORWARD = 0
    RIGHT_FORWARD = 1
    LEFT_UP = 2
    RIGHT_UP = 3

class ObjectSelectionStrategy(Enum):
    LARGEST = "largest"
    CLOSEST = "closest"
    LARGEST_CLOSEST = "largest_closest"

MAX_DEPTH = 3.8

class Vision:
    REAL_DIAMETER = 25
    FOCAL_LENGTH_PX = 400
    CENTER_X = 160
    CENTER_Y = 120

    hsv_ranges = {
        "RED": [
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255])
        ],
        "ORANGE": [([11, 50, 50], [20, 255, 255])],
        "YELLOW": [([21, 50, 50], [30, 255, 255])],
        "CHARTREUSE_GREEN": [([31, 50, 50], [40, 255, 255])],
        "GREEN": [([41, 50, 50], [70, 255, 255])],
        "SPRING_GREEN": [([71, 50, 50], [80, 255, 255])],
        "CYAN": [([81, 50, 50], [100, 255, 255])],
        "AZURE": [([101, 50, 50], [120, 255, 255])],
        "BLUE": [([121, 50, 50], [140, 255, 255])],
        "VIOLET": [([141, 50, 50], [150, 255, 255])],
        "MAGENTA": [([151, 50, 50], [160, 255, 255])],
        "ROSE": [([161, 50, 50], [179, 255, 255])],
    }

    @staticmethod
    def get_hsv_bounds(color: Color):
        return [(np.array(lower), np.array(upper)) for (lower, upper) in Vision.hsv_ranges[color.name]]


    @staticmethod
    def find_object(image, color: Color, strategy: ObjectSelectionStrategy = ObjectSelectionStrategy.LARGEST):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in Vision.get_hsv_bounds(color):
            mask = cv2.inRange(hsv, lower, upper)
            total_mask = cv2.bitwise_or(total_mask, mask)

        total_mask = cv2.GaussianBlur(total_mask, (7, 7), 2)

        contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = [c for c in contours if cv2.contourArea(c) > 200]
        if not contours:
            return None

        height, width = total_mask.shape
        image_center = (width / 2, height / 2)

        def distance_to_center(contour):
            (x, y), _ = cv2.minEnclosingCircle(contour)
            return (x - image_center[0]) ** 2 + (y - image_center[1]) ** 2

        if strategy == ObjectSelectionStrategy.LARGEST:
            selected = max(contours, key=cv2.contourArea)

        elif strategy == ObjectSelectionStrategy.CLOSEST:
            selected = min(contours, key=distance_to_center)

        elif strategy == ObjectSelectionStrategy.LARGEST_CLOSEST:
            largest_area = max(cv2.contourArea(c) for c in contours)
            big_enough = [c for c in contours if cv2.contourArea(c) >= 0.9 * largest_area]
            if not big_enough:
                logging.warning("No big enough contours found")
                return None

            selected = min(big_enough, key=distance_to_center)

        else:
            logging.info(f"Selected strategy 4")
            raise ValueError(f"Unknown selection strategy: {strategy}")

        (x, y), radius = cv2.minEnclosingCircle(selected)
        return (x, y), radius






    @staticmethod
    def analyze_object(image, color: Color, strategy: ObjectSelectionStrategy = ObjectSelectionStrategy.LARGEST):
        result = Vision.find_object(image, color, strategy)
        if result is None:
            return None

        (x, y), radius = result
        if radius < 5:
            return None

        distance = (Vision.REAL_DIAMETER * Vision.FOCAL_LENGTH_PX) / (radius * 2)
        return {"center": (x ,y), "radius": radius,"distance": distance}

class TaskManager:
    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.current_task = None
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def add_task(self, task):
        async with self.lock:
            await self.task_queue.put(task)
            logging.info(f"[Task Manager] Task added: {type(task).__name__}")

    async def run(self):
        while not self.stop_event.is_set():
            try:
                if self.current_task is None:
                    try:
                        self.current_task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                        logging.info(f"[TaskManager] Starting task: {type(self.current_task).__name__}")
                    except asyncio.TimeoutError:
                        continue

                await self.current_task.execute()

                if self.current_task.is_completed():
                    logging.info(f"[TaskManager] Task completed: {type(self.current_task).__name__}")
                    await self.current_task.motors.stop_all()
                    self.current_task = None
                    continue

            except asyncio.TimeoutError:
                logging.warning(f"[TaskManager] Task {type(self.current_task).__name__} timed out!")
                await self.current_task.motors.stop_all()
                self.current_task = None

            except Exception as e:
                logging.error(f"[TaskManager] Task crashed: {type(self.current_task).__name__} exception: {e}", exc_info=True)
                emergency_task = EmergencyStopTask()
                if self.current_task:
                    emergency_task.set_dependencies(
                        self.current_task.motors,
                        self.current_task.sensors,
                        self.current_task.camera
                    )
                await self.add_task(emergency_task)
                self.current_task = None

            await asyncio.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        if self.current_task:
            self.current_task.cancel()

class MotorController:
    def __init__(self):
        self.powers = [0, 0, 0, 0, 0]
        self.lock = asyncio.Lock()

    async def set_power(self, motor_id, power):
        async with self.lock:
            power = clamp(power, -100, 100)
            self.powers[int(motor_id)] = power
            auv.set_motor_power(motor_id, power)

    async def stop_all(self):
        for i in range(3):
            await self.set_power(i, 0)

class SensorHub:
    def __init__(self):
        self.depth_cache = 0.0
        self.yaw_cache = 0.0
        self.last_update = time.time()

    async def update(self):
        self.depth_cache = auv.get_depth()
        self.yaw_cache = auv.get_yaw()
        self.last_update = time.time()

    def get_depth(self):
        return self.depth_cache

    def get_yaw(self):
        return self.yaw_cache

    def is_fresh(self, max_age = 0.1):
        return (time.time() - self.last_update) < max_age

class CameraSystem:
    def __init__(self):
        self.last_bottom_image = None
        self.last_forward_image = None
        self.last_update = 0
        self.lock = asyncio.Lock()

    async def capture(self):
        async with self.lock:
            img = auv.get_image_bottom()
            if img is not None and img.size != 0:
                self.last_bottom_image = img.copy()
            else:
                logging.warning("Empty or invalid bottom image received")

            img = auv.get_image_front()
            if img is not None and img.size != 0:
                self.last_forward_image = img.copy()
            else:
                logging.warning("Empty or invalid forward image received")

        self.last_update = time.time()

    def get_bottom_image(self):
        return self.last_bottom_image.copy() if self.last_bottom_image is not None else None
    def get_forward_image(self):
        return self.last_forward_image.copy() if self.last_forward_image is not None else None

class Robot:
    def __init__(self):
        self.task_manager = TaskManager()
        self.motor = MotorController()
        self.sensor = SensorHub()
        self.camera = CameraSystem()

        self.sensor_update_task = None
        self.camera_update_task = None
        self.task_runner = None

    async def start(self):
        self.sensor_update_task = asyncio.create_task(self._update_sensors())
        self.camera_update_task = asyncio.create_task(self._update_camera())
        self.task_runner = asyncio.create_task(self.task_manager.run())

    async def _update_sensors(self):
        while not self.task_manager.stop_event.is_set():
            await self.sensor.update()
            await asyncio.sleep(0.05)

    async def _update_camera(self):
        while not self.task_manager.stop_event.is_set():
            await self.camera.capture()
            await asyncio.sleep(0.1)

    async def add_task(self, task):
        task.set_dependencies(self.motor, self.sensor, self.camera)
        await self.task_manager.add_task(task)

    async def stop(self):
        self.task_manager.stop()
        if self.sensor_update_task is not None:
            self.sensor_update_task.cancel()
            try:
                await self.sensor_update_task
            except asyncio.CancelledError:
                pass

        if self.camera_update_task is not None:
            self.camera_update_task.cancel()
            try:
                await self.camera_update_task
            except asyncio.CancelledError:
                pass

        await self.motor.stop_all()

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, setpoint, measurement):

        error = setpoint - measurement
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        P = self.kp * error

        self.integral += error * dt
        self.integral =  clamp(self.integral, -100, 100)
        I = self.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D = self.kd * derivative
        self.prev_error = error

        return P + I + D

class MovementTask(ABC):
    def set_dependencies(self, motors, sensors, camera):
        self.motors = motors
        self.sensors = sensors
        self.camera = camera

    @abstractmethod
    async def execute(self):
        pass

    @abstractmethod
    def is_completed(self) -> bool:
        pass

    def cancel(self):
        pass

class YawTask(MovementTask):
    def __init__(self, target_yaw, duration=4.0):
        self.target_yaw = target_yaw
        self.start_time = None
        self.duration = duration
        self.controller = PIDController(0.8, 0.01, 0.6)

    async def execute(self):
        if self.start_time is None:
            self.start_time = time.time()

        current_yaw = self.sensors.get_yaw()
        power = self.controller.compute(self.target_yaw, current_yaw)
        power = clamp_to_180(power)

        await self.motors.set_power(0, int(power))
        await self.motors.set_power(1, int(-power))

        await asyncio.sleep(0.05)

    def is_completed(self) -> bool:
        return (time.time() - self.start_time) >= self.duration if self.start_time else False

class DepthTask(MovementTask):
    def __init__(self, target_depth, duration=10.0):
        self.target_depth = target_depth
        self.start_time = None
        self.duration = duration
        self.controller = PIDController(80, 0.2, 65)

    async def execute(self):
        if self.start_time is None:
            self.start_time = time.time()

        current_depth = self.sensors.get_depth()
        power = self.controller.compute(self.target_depth, current_depth)
        power = clamp(power, -100, 100)

        await self.motors.set_power(int(MotorNums.LEFT_UP), int(-power))
        await self.motors.set_power(int(MotorNums.RIGHT_UP), int(-power))

        await asyncio.sleep(0.05)

    def is_completed(self) -> bool:
        return (time.time() - self.start_time) >= self.duration if self.start_time else False

class AlignToObjectTask(MovementTask):
    def __init__(self, color: Color, strategy: ObjectSelectionStrategy = ObjectSelectionStrategy.LARGEST, timeout=30.0, tolerance=10):
        self.color = color
        self.timeout = timeout
        self.tolerance = tolerance
        self.strategy = strategy
        self.start_time = None
        self.last_aligned_time = None

        self.pid_yaw = PIDController(0.4, 0.0, 0.2)
        self.pid_forward = PIDController(0.4, 0.0, 0.2)

    async def execute(self):
        if self.start_time is None:
            self.start_time = time.time()

        image = self.camera.get_bottom_image()
        result = Vision.analyze_object(image, self.color, self.strategy)

        if result is None:
            logging.debug("Can't find center of object")
            await self.motors.set_power(0, 0)
            await self.motors.set_power(1, 0)
            return

        x, y = result.get("center")

        yaw_power = int(np.clip(self.pid_yaw.compute(Vision.CENTER_X, x), -100, 100))
        forward_power = int(np.clip(self.pid_forward.compute(Vision.CENTER_Y, y), -100, 100))

        await self.motors.set_power(0, forward_power - yaw_power)
        await self.motors.set_power(1, forward_power + yaw_power)

        if (abs(Vision.CENTER_X - x) < self.tolerance) and (abs(Vision.CENTER_Y - y) < self.tolerance):
            if self.last_aligned_time is None:
                self.last_aligned_time = time.time()
        else:
            self.last_aligned_time = None

        await asyncio.sleep(0.05)

    def is_completed(self) -> bool:
        current_time = time.time()

        if self.last_aligned_time is None:
            return False

        if (current_time - self.last_aligned_time) >= self.timeout:
            logging.info("Aligning timeout")
            return True

        if (current_time - self.last_aligned_time) >= 3:
            logging.info("Object aligned")
            return True

        return False

class DescendToObject(MovementTask):
    def __init__(self, desired_distance, color:Color, strategy: ObjectSelectionStrategy = ObjectSelectionStrategy.LARGEST,duration=10.0):
        self.desired_distance = desired_distance
        self.duration = duration
        self.color = color
        self.strategy = strategy
        self.depth_task = None
        self.started = False
        self.start_time = None

    async def execute(self):
        if self.start_time is None:
            self.start_time = time.time()

        if self.depth_task is None and not self.started:
            self.started = True

            image = self.camera.get_bottom_image()
            result = Vision.analyze_object(image, self.color, self.strategy)

            if result is None:
                logging.warning("Can't find object")
                return

            current_distance = result["distance"] / 100
            current_depth = self.sensors.get_depth()

            target_depth = MAX_DEPTH - self.desired_distance

            logging.info(f"Current distance: {current_distance:.2f}m, "
                         f"Current depth: {current_depth:.2f}m, "
                         f"Target depth: {target_depth:.2f}m")

            self.depth_task = DepthTask(target_depth, self.duration)
            self.depth_task.set_dependencies(self.motors, self.sensors, self.camera)

        if self.depth_task is not None:
            await self.depth_task.execute()

        await asyncio.sleep(0.05)

    def is_completed(self) -> bool:
        if self.depth_task is not None:
            return self.depth_task.is_completed()

        if self.start_time and (time.time() - self.start_time) > self.duration:
            logging.warning("DescendToObject timeout waiting for object")
            return True

        return False

    def cancel(self):
        if self.depth_task:
            self.depth_task.cancel()

class YawAlignToObject(MovementTask):
    def __init__(self, color: Color, front: bool = False, strategy: ObjectSelectionStrategy = ObjectSelectionStrategy.LARGEST, tolerance=10, timeout=10):
        self.color = color
        self.tolerance = tolerance
        self.timeout = timeout
        self.strategy = strategy
        self.start_time = None
        self.front = front
        self.last_aligned_time = None
        self.controller = PIDController(0.4, 0.0, 0.2)

    async def execute(self):
        if self.start_time is None:
            self.start_time = time.time()


        image = self.camera.get_forward_image() if self.front  else self.camera.get_bottom_image()
        result = Vision.analyze_object(image, self.color, self.strategy)

        if result is None:
            await self.motors.stop_all()
            return

        x, _ = result.get("center")
        power = self.controller.compute(Vision.CENTER_X, x)
        power = clamp(power, -100, 100)

        await self.motors.set_power(int(MotorNums.LEFT_FORWARD), -int(power))
        await self.motors.set_power(int(MotorNums.RIGHT_FORWARD), int(power))

        if abs(Vision.CENTER_X - x) < self.tolerance:
            if self.last_aligned_time is None:
                self.last_aligned_time = time.time()
            else:
                self.last_aligned_time = None

        await asyncio.sleep(0.05)

    def is_completed(self) -> bool:
        if self.last_aligned_time is None:
            return False

        if (time.time() - self.last_aligned_time) >= 3:
            logging.info(f"Yaw aligned to {self.color.name}")
            return True

        if self.start_time >= self.timeout:
            logging.warning(f"Yaw aligned timeout waiting for {self.color.name}")
            return True
        return False

class DepthHoldController:
    def __init__(self, auv, target_depth):
        self.auv = auv
        self.target_depth = target_depth
        self.pid = PIDController(kp=80, ki=0.5, kd=45)
        self.running = True

    async def run(self):
        while self.running:
            depth = self.auv.get_depth()
            power = self.pid.compute(self.target_depth, depth)

            self.auv.set_motor_power(int(MotorNums.RIGHT_UP), -int(power))
            self.auv.set_motor_power(int(MotorNums.LEFT_UP), -int(power))
            await asyncio.sleep(0.05)

    def stop(self):
        self.running = False

class YawHoldController:
    def __init__(self, auv, target_yaw):
        self.auv = auv
        self.target_yaw = target_yaw
        self.pid = PIDController(kp=1.5, ki=0.01, kd=0.2)
        self.running = True

    async def run(self):
        while self.running:
            current_yaw = self.auv.get_yaw()
            error = (self.target_yaw - current_yaw + 180) % 360 - 180
            power = self.pid.compute(0, error)
            self.auv.set_motor_power(int(MotorNums.RIGHT_FORWARD), int(power))
            self.auv.set_motor_power(int(MotorNums.LEFT_FORWARD), -int(power))
            await asyncio.sleep(0.05)

    def stop(self):
        self.running = False

class CompositeTask(MovementTask):
    def __init__(self, tasks):
        self.tasks = tasks
        self._cancelled = False

    def set_dependencies(self, motors, sensors, camera):
        self.motors = motors
        self.sensors = sensors
        self.camera = camera
        for t in self.tasks:
            t.set_dependencies(motors, sensors, camera)

    async def execute(self):
        async def run_task(task):
            while not task.is_completed():
                if self._cancelled:
                    task.cancel()
                    break
                await task.execute()

        await asyncio.gather(*(run_task(t) for t in self.tasks))

    def is_completed(self) -> bool:
        return all(t.is_completed() for t in self.tasks)

    def cancel(self):
        self._cancelled = True
        for t in self.tasks:
            t.cancel()

class EmergencyStopTask(MovementTask):
    async def execute(self):
        try:
            await self.motors.stop_all()
            logging.warning("EMERGENCY STOP ACTIVATED")
        except Exception as e:
            logging.error(f"Emergency stop failed: {e}")
            raise

    def is_completed(self) -> bool:
        return True

def clamp_to_180(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def clamp(v, min, max):
    if v < min:
        return min
    if v > max:
        return max
    return v

async def main():

    robot = Robot()
    try:
        await robot.camera.capture()
        await robot.sensor.update()

        if robot.camera.get_bottom_image() is None:
            raise RuntimeError("Camera not working")
        if not robot.sensor.is_fresh():
            raise RuntimeError("Sensor not working")

        await robot.start()
        await asyncio.sleep(0.5)

        depth_controller = DepthHoldController(auv, target_depth=1.5)
        yaw_controller = YawHoldController(auv, target_yaw=90.0)

        depth_task = asyncio.create_task(depth_controller.run())
        #yaw_task = asyncio.create_task(yaw_controller.run())
        await robot.add_task(AlignToObjectTask(color=Color.RED, strategy=ObjectSelectionStrategy.LARGEST_CLOSEST, timeout=10, tolerance=2))
        #await robot.add_task(DepthTask(target_depth=2, duration=10.0))

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    finally:
        await robot.stop()

if __name__ == "__main__":
    asyncio.run(main())

