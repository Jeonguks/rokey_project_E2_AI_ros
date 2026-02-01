#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import Bool

from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
from ultralytics import YOLO

import numpy as np
import cv2
import threading
import math
import time as pytime

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions


class FollowCarAfterTrigger(Node):
    """
    - /object_detected(True) -> waypoint로 이동(goToPose)
    - 도착 후 YOLO로 'car' 탐지
    - depth + intrinsics + TF로 car의 map 좌표 계산
    - car 방향으로 접근하되 stop_distance 만큼 떨어진 지점으로 이동(goToPose)
    - MultiThreadedExecutor + GUI thread
    """

    WAIT_TRIGGER = 0
    GO_WAYPOINT = 1
    SEARCH_CAR = 2
    GO_CAR_FRONT = 3
    DONE = 4

    def __init__(self):
        super().__init__('follow_car_after_trigger')

        #########################################
        # PARAMETERS  (유지)
        #########################################
        self.trigger_topic = "/object_detected"

        self.rgb_topic = "/robot2/oakd/rgb/preview/image_raw"
        self.depth_topic = "/robot2/oakd/stereo/image_raw"
        self.info_topic = "/robot2/oakd/rgb/preview/camera_info"

        self.map_frame = "map"
        self.base_frame = "base_link"   # base_link만 사용

        self.waypoint_xy = (-1.88335, 1.41718)   # 트리거 후 1차 이동 지점 (map)
        self.waypoint_yaw = -2.36         # rad   z: -0.9212606138352417 w: 0.3889458592091111

        self.target_class = "car"
        self.yolo_weights = "/home/rokey/Downloads/amr_default_best.pt"

        self.stop_distance = 0.5        # car 앞에서 멈출 거리(m)
        self.loop_hz = 5.0              # 메인 tick 주기
        self.goal_interval = 1.5        # goal 폭주 방지(초)

        # -------------------------

        # 상태
        self.state = self.WAIT_TRIGGER
        self.sent_waypoint = False
        self.sent_car_goal = False
        self.last_goal_time = 0.0

        # 센서/연산 상태
        self.bridge = CvBridge()
        self.K = None
        self.rgb = None
        self.depth = None
        self.camera_frame = None
        self.lock = threading.Lock()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLO
        self.model = YOLO(self.yolo_weights)

        # Navigator
        self.navigator = TurtleBot4Navigator()

        # 초기 pose 세팅 + Nav2 active 대기 (DepthToMap 방식)
        initial_pose = self.navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()

        # ROS Sub
        # self.create_subscription(Bool, self.trigger_topic, self.on_trigger, 10)  # ✅ 주석 처리
        self.create_subscription(CameraInfo, self.info_topic, self.on_info, 10)
        self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

        # ✅ 조건문 부분 일단 True로: 시작하자마자 GO_WAYPOINT로 진입
        self.state = self.GO_WAYPOINT
        self.sent_waypoint = False
        self.sent_car_goal = False

        # GUI
        self.display_frame = None
        self.gui_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        # Main tick
        self.timer = self.create_timer(1.0 / self.loop_hz, self.tick)

        self.get_logger().info("FollowCarAfterTrigger started.")
        self.get_logger().info(f"(trigger forced TRUE) -> start at GO_WAYPOINT")
        self.get_logger().info(f"waypoint: {self.waypoint_xy} yaw={self.waypoint_yaw:.2f}")
        self.get_logger().info(f"stop_distance: {self.stop_distance:.2f} m")

    # -------------------------
    # Callbacks (유지 스타일)
    # -------------------------
    def on_trigger(self, msg: Bool):
        if bool(msg.data) and self.state == self.WAIT_TRIGGER:
            self.get_logger().info("[STATE] Trigger=True -> GO_WAYPOINT")
            self.state = self.GO_WAYPOINT
            self.sent_waypoint = False
            self.sent_car_goal = False

    def on_info(self, msg: CameraInfo):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_rgb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.rgb = img
                self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"RGB convert failed: {e}")

    def on_depth(self, msg: Image):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.depth = d
        except Exception as e:
            self.get_logger().warn(f"Depth convert failed: {e}")

    # -------------------------
    # Main loop
    # -------------------------
    def tick(self):
        if self.state == self.WAIT_TRIGGER:
            return

        # goal 폭주 방지(DepthToMap / 기존 코드 스타일)
        now = pytime.time()
        can_send_goal = (now - self.last_goal_time) >= self.goal_interval

        if self.state == self.GO_WAYPOINT:
            if (not self.sent_waypoint) and can_send_goal:
                goal_pose = PoseStamped()
                goal_pose.header.frame_id = self.map_frame
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = float(self.waypoint_xy[0])
                goal_pose.pose.position.y = float(self.waypoint_xy[1])
                goal_pose.pose.position.z = 0.0

                qz = math.sin(self.waypoint_yaw / 2.0)
                qw = math.cos(self.waypoint_yaw / 2.0)
                goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

                self.get_logger().info(f"[GO] waypoint ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})")
                self.navigator.goToPose(goal_pose)

                self.sent_waypoint = True
                self.last_goal_time = now
                return

            if self.sent_waypoint and self.navigator.isTaskComplete():
                self.get_logger().info("[STATE] Waypoint reached -> SEARCH_CAR")
                self.state = self.SEARCH_CAR
            return

        if self.state == self.SEARCH_CAR:
            with self.lock:
                rgb = self.rgb.copy() if self.rgb is not None else None
                depth = self.depth.copy() if self.depth is not None else None
                K = self.K.copy() if self.K is not None else None
                cam_frame = self.camera_frame

            if rgb is None or depth is None or K is None or cam_frame is None:
                self.get_logger().warn(f"[WARN] rgb: {rgb} depth: {depth} K:{K}, cam_frame:{cam_frame}")
                return

            # YOLO (YoloPersonNavGoal 방식)
            try:
                res = self.model(rgb, verbose=False)[0]
            except Exception as e:
                self.get_logger().warn(f"YOLO failed: {e}")
                return

            frame = rgb.copy()

            best = None
            best_conf = -1.0
            for det in res.boxes:
                cls = int(det.cls[0])
                label = self.model.names[cls]
                conf = float(det.conf[0])
                if label.lower() == self.target_class and conf > best_conf:
                    best = det
                    best_conf = conf

            if best is None:
                cv2.putText(frame, "STATE: SEARCH_CAR (no car)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.display_frame = frame
                return

            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            u = int((x1 + x2) // 2)
            v = int((y1 + y2) // 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"car {best_conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            h, w = depth.shape[:2]
            u = min(max(u, 0), w - 1)
            v = min(max(v, 0), h - 1)

            z = float(depth[v, u]) / 1000.0
            if z <= 0.01 or math.isnan(z) or math.isinf(z):
                cv2.putText(frame, "depth invalid", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.display_frame = frame
                return

            cv2.putText(frame, f"depth={z:.2f}m", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z

            pt = PointStamped()
            pt.header.frame_id = cam_frame
            pt.header.stamp = Time().to_msg()
            pt.point.x = float(X)
            pt.point.y = float(Y)
            pt.point.z = float(Z)

            # camera -> map (DepthToMap / YoloPersonNavGoal 방식: tf_buffer.transform)
            try:
                pt_map = self.tf_buffer.transform(pt, self.map_frame, timeout=Duration(seconds=0.5))
            except Exception as e:
                self.get_logger().warn(f"TF transform to map failed: {e}")
                self.display_frame = frame
                return

            car_x, car_y = pt_map.point.x, pt_map.point.y
            self.get_logger().info(f"[CAR] map ({car_x:.2f}, {car_y:.2f}, {pt_map.point.z:.2f})")

            # 로봇(map) 좌표: base_link 원점을 map으로 transform (tf_buffer.transform만 사용)
            base_pt = PointStamped()
            base_pt.header.frame_id = self.base_frame
            base_pt.header.stamp = Time().to_msg()
            base_pt.point.x = 0.0
            base_pt.point.y = 0.0
            base_pt.point.z = 0.0

            try:
                base_map = self.tf_buffer.transform(base_pt, self.map_frame, timeout=Duration(seconds=0.5))
            except Exception as e:
                self.get_logger().warn(f"TF base_link->map failed: {e}")
                self.display_frame = frame
                return

            rx, ry = base_map.point.x, base_map.point.y

            vx = car_x - rx
            vy = car_y - ry
            dist = math.hypot(vx, vy)
            if dist < 1e-6:
                self.display_frame = frame
                return

            ux = vx / dist
            uy = vy / dist

            gx = car_x - ux * self.stop_distance
            gy = car_y - uy * self.stop_distance
            gyaw = math.atan2(car_y - gy, car_x - gx)

            self.car_goal = (gx, gy, gyaw)

            cv2.putText(frame, "STATE: CAR_FOUND -> GO_CAR_FRONT", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.display_frame = frame
            self.state = self.GO_CAR_FRONT
            self.sent_car_goal = False
            return

        if self.state == self.GO_CAR_FRONT:
            with self.lock:
                frame = self.rgb.copy() if self.rgb is not None else None

            if frame is None:
                return

            cv2.putText(frame, "STATE: GO_CAR_FRONT", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if (not self.sent_car_goal) and hasattr(self, "car_goal") and can_send_goal:
                gx, gy, gyaw = self.car_goal

                goal_pose = PoseStamped()
                goal_pose.header.frame_id = self.map_frame
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.pose.position.x = float(gx)
                goal_pose.pose.position.y = float(gy)
                goal_pose.pose.position.z = 0.0

                qz = math.sin(gyaw / 2.0)
                qw = math.cos(gyaw / 2.0)
                goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

                self.get_logger().info(f"[GO] car-front ({gx:.2f}, {gy:.2f}) yaw={gyaw:.2f}")
                self.navigator.goToPose(goal_pose)

                self.sent_car_goal = True
                self.last_goal_time = now

            if self.sent_car_goal and self.navigator.isTaskComplete():
                self.get_logger().info("[DONE] reached car-front")
                self.state = self.DONE

            self.display_frame = frame
            return

        if self.state == self.DONE:
            with self.lock:
                frame = self.rgb.copy() if self.rgb is not None else None
            if frame is not None:
                cv2.putText(frame, "STATE: DONE", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                self.display_frame = frame
            return

    # -------------------------
    # GUI loop (유지 스타일)
    # -------------------------
    def gui_loop(self):
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection", 960, 540)

        while not self.gui_stop.is_set() and rclpy.ok():
            img = self.display_frame.copy() if self.display_frame is not None else None
            if img is not None:
                cv2.imshow("YOLO Detection", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.get_logger().info("GUI shutdown.")
                    # self.gui_stop.set()
                    # rclpy.shutdown()
                    break
                elif key == ord('r'):
                    self.get_logger().info("Reset -> SEARCH_CAR")
                    self.state = self.SEARCH_CAR
                    self.sent_car_goal = False
            else:
                cv2.waitKey(10)

        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = FollowCarAfterTrigger()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.gui_stop.set()
    node.destroy_node()
    cv2.destroyAllWindows()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
