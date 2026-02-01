import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO
import threading
import time
import cv2
import math

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions


class YoloPersonNavGoal(Node):

    WAIT_TRIGGER = 0
    GO_WAYPOINT = 1
    SEARCH_CAR = 2
    GO_CAR_FRONT = 3
    DONE = 4

    def __init__(self):
        super().__init__('nav_to_person')


        # Initialize Trigger State 
        self.state = self.WAIT_TRIGGER
        self.sent_waypoint = False
        self.sent_car_goal = False

        # Internal state
        self.dock_requested = False #  도킹 명령 중복전송 방지
        self.undock_requested = False

        self.waypoint_x = -1.88335
        self.waypoint_y = 1.41718
        self.waypoint_yaw = -2.36

        self.bridge = CvBridge()
        self.K = None
        self.depth_image = None
        self.rgb_image = None
        self.camera_frame = None
        self.latest_map_point = None

        self.goal_handle = None
        self.shutdown_requested = False
        self.logged_intrinsics = False
        self.current_distance = None
        self.close_enough_distance = 2.0
        self.block_goal_updates = False
        self.close_distance_hit_count = 0

        self.model = YOLO("/home/jeonguk/Downloads/yolov8n.pt")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.display_frame = None
        self.display_thread = threading.Thread(
            target=self.display_loop,
            daemon=True
        )
        self.display_thread.start()

        self.create_subscription(CameraInfo,'/oakd/rgb/preview/camera_info',self.camera_info_callback,10)
        self.create_subscription(Image,'/oakd/rgb/preview/image_raw',self.rgb_callback,10)
        self.create_subscription(Image,'/oakd/rgb/preview/depth',self.depth_callback,10)
        self.create_subscription(Bool, '/object_detected', self.on_trigger_callback, 10)


        self.last_feedback_log_time = 0

        self.navigator = TurtleBot4Navigator()

        if (not self.navigator.getDockedStatus()) and (not self.dock_requested):
            self.dock_requested = True
            self.get_logger().info("Dock requested")
            self.navigator.dock()
            self.dock_requested = not self.navigator.getDockedStatus() # 도킹이면 false로 플래그 설정


        initial_pose = self.navigator.getPoseStamped(
            [0.0, 0.0],
            TurtleBot4Directions.NORTH
        )
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("초기화 완료 , 순찰 명령 까지 대기")


        self.state_enter_time = time.time()
        self.create_timer(0.2, self.main_loop)

    def _enter_state(self, new_state: int):
        self.state = new_state
        self.state_enter_time = time.time()

        if new_state == self.GO_WAYPOINT and not self.undock_requested:
            self.undock_requested = True
            self.sent_waypoint = False
            self.sent_car_goal = False
            self.undock_requested = self.navigator.getDockedStatus() # 언도킹이면 false로 플래그 설정

        if new_state == self.SEARCH_CAR:
            self.sent_car_goal = False
            if self.navigator.isTaskComplete() is False:
                pass

        if new_state == self.GO_CAR_FRONT:
            self.sent_car_goal = False

        self.get_logger().info(f"[STATE] -> {new_state}")

    def on_trigger_callback(self, msg: Bool):
        if msg.data and self.state == self.WAIT_TRIGGER:
            self.get_logger().info(
                "[TRIGGER] object_detected=True -> GO_WAYPOINT"
            )
            self._enter_state(self.GO_WAYPOINT)

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)
        if not self.logged_intrinsics:
            self.get_logger().info(
                f"Camera intrinsics received: "
                f"fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
            )
            self.logged_intrinsics = True

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
            self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def process_frame(self):
        if self.K is None or self.rgb_image is None or self.depth_image is None:
            return

        results = self.model(self.rgb_image, verbose=False)[0]
        frame = self.rgb_image.copy()

        for det in results.boxes:
            cls = int(det.cls[0])
            label = self.model.names[cls]
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(
                int,
                det.xyxy[0].tolist()
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            if label.lower() == "person":
                u = int((x1 + x2) // 2)
                v = int((y1 + y2) // 2)
                z = float(self.depth_image[v, u])

                if z == 0.0:
                    self.get_logger().warn(
                        "Depth value is 0 at detected person's center."
                    )
                    continue

                fx, fy = self.K[0, 0], self.K[1, 1]
                cx, cy = self.K[0, 2], self.K[1, 2]
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                pt = PointStamped()
                pt.header.frame_id = self.camera_frame
                pt.header.stamp = rclpy.time.Time().to_msg()
                pt.point.x = x
                pt.point.y = y
                pt.point.z = z

                try:
                    pt_map = self.tf_buffer.transform(
                        pt,
                        'map',
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    self.latest_map_point = pt_map

                    if self.block_goal_updates:
                        break

                    if self.goal_handle:
                        self.goal_handle.cancel_goal_async()

                    self.send_goal(
                        self.latest_map_point.point.x,
                        self.latest_map_point.point.y,
                        yaw=0.0
                    )

                except Exception as e:
                    self.get_logger().warn(
                        f"TF transform to map failed: {e}"
                    )
                break

        self.display_frame = frame

    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=qz,
            w=qw
        )

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.action_client.wait_for_server()
        self._send_goal_future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    def feedback_callback(self, feedback_msg):
        self.current_distance = feedback_msg.feedback.distance_remaining

        if (
            self.current_distance is not None
            and self.current_distance < self.close_enough_distance
        ):
            self.close_distance_hit_count += 1
        else:
            self.close_distance_hit_count = 0

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            return
        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(
            self.goal_result_callback
        )

    def goal_result_callback(self, future):
        self.goal_handle = None

    def display_loop(self):
        while rclpy.ok():
            if self.display_frame is not None:
                cv2.imshow("YOLO Detection", self.display_frame)
                if cv2.waitKey(1) == 27:
                    self.shutdown_requested = True
                    break
            time.sleep(0.01)

    def main_loop(self):
        if self.state == self.WAIT_TRIGGER:
            return

        if self.state == self.GO_WAYPOINT:
            if not self.sent_waypoint:
                self.navigator.undock()
                self.send_goal(
                    self.waypoint_x,
                    self.waypoint_y,
                    self.waypoint_yaw
                )
                self.sent_waypoint = True
            return
        
        if self.state == self.SEARCH_CAR:
            self.create_timer(0.5, self.process_frame)

        if self.state == self.GO_CAR_FRONT:
            self.create_timer(0.5, self.process_frame)

        if self.state == self.DONE:
            self.send_goal(0,0)


def main():
    rclpy.init()
    node = YoloPersonNavGoal()

    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()