import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion

from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

from ultralytics import YOLO

import numpy as np
import cv2
import threading
import math


class DepthToMap(Node):
    def __init__(self):
        super().__init__('depth_to_map_node')

        self.bridge = CvBridge()
        self.K = None
        self.lock = threading.Lock()

        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'

        self.depth_image = None
        self.rgb_image = None
        self.clicked_point = None
        self.shutdown_requested = False
        self.display_image = None


        # Load YOLOv8 model ################################################
        self.model = YOLO("//home/rokey/Downloads/amr_default_best.pt")
        ####################################################################


        self.gui_thread_stop = threading.Event()
        self.gui_thread = threading.Thread(target=self.gui_loop, daemon=True)
        self.gui_thread.start()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = TurtleBot4Navigator()



        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()

        initial_pose = self.navigator.getPoseStamped([0.0, 0.0], TurtleBot4Directions.NORTH)
        initial_detection_pose = self.navigator.getPoseStamped([-1.88335,1.41718], TurtleBot4Directions.SOUTH_EAST) # 잘 보이는 위치로 이동 

        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.navigator.undock()

        self.logged_intrinsics = False
        self.logged_rgb_shape = False
        self.logged_depth_shape = False

        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 1)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 1)
        self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, 1)

        self.get_logger().info("TF Tree 안정화 시작. 5초 후 변환 시작합니다.")
        self.start_timer = self.create_timer(5.0, self.start_transform)

        self.navigator.startToPose(initial_detection_pose)


        # # --- initial pose / docking 관련 전부 스킵 ---
        # initial_detection_pose = self.navigator.getPoseStamped(
        #     [-1.88335, 1.41718],
        #     TurtleBot4Directions.SOUTH_EAST
        # )

        # Nav2만 활성화 대기 (localization이 이미 준비돼있다는 전제)
        self.navigator.waitUntilNav2Active()

        # 바로 목적지로 이동
        self.navigator.startToPose(initial_detection_pose)

        self.enable_yolo_overlay = True
        self.yolo_conf_th = 0.25
        self.yolo_target_label = "car"   # 필요하면 None으로 해서 전부 그리기
        self.last_yolo_frame = None


        # --- goal update control ---
        self.yolo_center_pixel = None
        self.yolo_center_conf = 0.0
        self.last_goal_time = self.get_clock().now()
        self.goal_period = 1.0  # 1초에 한번만 goal 보내기

#################################TODO FIX ############ 
    def draw_yolo_on_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """RGB 프레임에 YOLO bbox를 그려서 반환"""
        if rgb is None:
            return None

        # Ultralytics는 BGR도 잘 돌지만, 혹시 색 이상하면 cvtColor로 RGB->BGR 조정 가능
        results = self.model(rgb, verbose=False)[0]
        out = rgb.copy()
        best_center = None
        best_conf = -1.0


        for det in results.boxes:
            conf = float(det.conf[0])
            if conf < self.yolo_conf_th:
                continue

            cls = int(det.cls[0])
            label = self.model.names[cls]

            if self.yolo_target_label and label.lower() != self.yolo_target_label.lower():
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            # 바운딩 박스 그리기 
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # center candidate
            u = int((x1 + x2) // 2)
            v = int((y1 + y2) // 2)

            if conf > best_conf:
                best_conf = conf
                best_center = (u, v)


                #  센터 픽셀 저장(그리기 함수에서 뽑아내기)
        with self.lock:
            self.yolo_center_pixel = best_center
            self.yolo_center_conf = best_conf
        if best_center is not None:
            self.get_logger().info(
                f"[YOLO] center_pixel = (u={best_center[0]}, v={best_center[1]}), conf={best_conf:.3f}"
            )
        # 시각화용 점
        if best_center is not None:
            cv2.circle(out, best_center, 5, (0, 0, 255), -1)
        
        return out


    def start_transform(self):
        self.get_logger().info("TF Tree 안정화 완료. 변환 시작합니다.")
        self.timer = self.create_timer(0.2, self.display_images)
        self.start_timer.cancel()

    def camera_info_callback(self, msg):
        with self.lock:
            self.K = np.array(msg.k).reshape(3, 3)
            if not self.logged_intrinsics:
                self.get_logger().info(
                    f"Camera intrinsics received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
                self.logged_intrinsics = True

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                if not self.logged_depth_shape:
                    self.get_logger().info(f"Depth image received: {depth.shape}")
                    self.logged_depth_shape = True
                with self.lock:
                    self.depth_image = depth
                    self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Depth CV bridge conversion failed: {e}")

    def rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is not None and rgb.size > 0:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f"RGB image decoded: {rgb.shape}")
                    self.logged_rgb_shape = True
                with self.lock:
                    self.rgb_image = rgb
        except Exception as e:
            self.get_logger().error(f"Compressed RGB decode failed: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:
                self.clicked_point = (x, y)
            self.get_logger().info(f"Clicked RGB pixel: ({x}, {y})")

    def process_frame(self):
        if self.K is None or self.rgb_image is None or self.depth_image is None:
            return

        results = self.model(self.rgb_image, verbose=False)[0]
        frame = self.rgb_image.copy()

        for det in results.boxes:
            cls = int(det.cls[0])
            label = self.model.names[cls]
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if label.lower() == "car":
                u = int((x1 + x2) // 2)
                v = int((y1 + y2) // 2)
                z = float(self.depth_image[v, u])

                if z == 0.0:
                    self.get_logger().warn("Depth value is 0 at detected car's center.")
                    continue

                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                pt = PointStamped()
                pt.header.frame_id = self.camera_frame
                pt.header.stamp = rclpy.time.Time().to_msg()
                pt.point.x, pt.point.y, pt.point.z = x, y, z

                try:
                    pt_map = self.tf_buffer.transform(pt, 'map', timeout=rclpy.duration.Duration(seconds=0.5))
                    self.latest_map_point = pt_map

                    # Don't send more goals if we're already close
                    if self.block_goal_updates:
                        self.get_logger().info(f"Within ({self.close_enough_distance}) meter — skipping further goal updates.")
                        break

                    self.get_logger().info(f"Detected car at map: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f})")

                    if self.goal_handle:
                        self.get_logger().info("Canceling previous goal...")
                        self.goal_handle.cancel_goal_async()

                    self.send_goal()

                except Exception as e:
                    self.get_logger().warn(f"TF transform to map failed: {e}")
                break

        self.display_frame = frame


    def display_images(self):
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            click = self.clicked_point
            frame_id = getattr(self, 'camera_frame', None)

        if rgb is not None and depth is not None and frame_id:
            try:
                # rgb_display = rgb.copy()
                # RGB 왼쪽 화면에 YOLO bbox overlay
                if self.enable_yolo_overlay:
                    try:
                        rgb_display = self.draw_yolo_on_rgb(rgb)
                        with self.lock:
                            self.last_yolo_frame = rgb_display.copy()
                    except Exception as e:
                        self.get_logger().warn(f"YOLO overlay failed: {e}")
                        rgb_display = rgb.copy()
                else:
                    rgb_display = rgb.copy()

                depth_display = depth.copy()
                depth_normalized = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                if click is not None:
                    x, y = click
                    z = float(depth[y, x]) / 1000.0
                    if 0.2 < z < 5.0:
                        fx, fy = self.K[0, 0], self.K[1, 1]
                        cx, cy = self.K[0, 2], self.K[1, 2]

                        X = (x - cx) * z / fx
                        Y = (y - cy) * z / fy
                        Z = z

                        pt_camera = PointStamped()
                        pt_camera.header.stamp = Time().to_msg()
                        pt_camera.header.frame_id = frame_id
                        pt_camera.point.x = X
                        pt_camera.point.y = Y
                        pt_camera.point.z = Z

                        pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=Duration(seconds=1.0))
                        self.get_logger().info(f"Map coordinate: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}, {pt_map.point.z:.2f})")

                        goal_pose = PoseStamped()
                        goal_pose.header.frame_id = 'map'
                        goal_pose.header.stamp = self.get_clock().now().to_msg()
                        goal_pose.pose.position.x = pt_map.point.x
                        goal_pose.pose.position.y = pt_map.point.y
                        goal_pose.pose.position.z = 0.0
                        yaw = 0.0
                        qz = math.sin(yaw / 2.0)
                        qw = math.cos(yaw / 2.0)
                        goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

                        self.navigator.goToPose(goal_pose)
                        self.get_logger().info("Sent navigation goal to map coordinate.")

                        with self.lock:
                            self.clicked_point = None

                    cv2.circle(rgb_display, (x, y), 4, (0, 255, 0), -1)
                    text = f"{z:.2f} m" if 0.2 < z < 5.0 else "Invalid"
                    cv2.putText(depth_colored, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.circle(depth_colored, (x, y), 4, (255, 255, 255), -1)

                combined = np.hstack((rgb_display, depth_colored))
                with self.lock:
                    self.display_image = combined.copy()
            except Exception as e:
                self.get_logger().warn(f"TF or goal error: {e}")

    def gui_loop(self):
        cv2.namedWindow('RGB (left) | Depth (right)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB (left) | Depth (right)', 1280, 480)
        cv2.moveWindow('RGB (left) | Depth (right)', 100, 100)
        cv2.setMouseCallback('RGB (left) | Depth (right)', self.mouse_callback)

        while not self.gui_thread_stop.is_set():
            with self.lock:
                img = self.display_image.copy() if self.display_image is not None else None

            if img is not None:
                cv2.imshow('RGB (left) | Depth (right)', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.get_logger().info("Shutdown requested by user (via GUI).")
                    self.navigator.dock()
                    self.shutdown_requested = True
                    self.gui_thread_stop.set()
                    rclpy.shutdown()
            else:
                cv2.waitKey(10)


def main():
    rclpy.init()
    node = DepthToMap()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.gui_thread_stop.set()
    node.gui_thread.join()
    node.destroy_node()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# This code is a ROS2 node that subscribes to depth and RGB images, allows the user to click on a point in the RGB image,
# and then calculates the corresponding 3D point in the map frame. It also sends a navigation goal to that point.
# The node uses a GUI to display the images and allows interaction via mouse clicks.
# It also handles TF transformations to convert the clicked point from the camera frame to the map frame.
# The node is designed to run in a multi-threaded executor to handle image processing and GUI updates concurrently.
# The code includes error handling for TF transformations and image processing, ensuring robustness in various scenarios.
# The node also logs important information such as camera intrinsics and image shapes for debugging purposes.