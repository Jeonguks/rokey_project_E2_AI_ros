#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration

class BeepNode(Node):
	def **init**(self):
	super().**init**('turtlebot4_beep_node')

	    topic = '/robot2/cmd_audio'

	    self.pub = self.create_publisher(AudioNoteVector, topic, 10)

	    msg = AudioNoteVector()
	    msg.append = False

	    def note(freq, sec=0, nsec=300_000_000):
		n = AudioNote()
		n.frequency = float(freq)
		n.max_runtime = Duration(sec=sec, nanosec=nsec)
		return n

	    # 삐뽀삐뽀
	    msg.notes = [
		note(880),  
		note(440),  
		note(880),  
		note(440),  
	    ]

	    self.get_logger().info(f'Sending BEEP sequence to {topic}')
	    self.pub.publish(msg)

	    # 한번 보내고 종료
	    rclpy.shutdown()

	```

	def main():
	rclpy.init()
	node = BeepNode()
	rclpy.spin_once(node, timeout_sec=0.2)

	if **name** == '**main**':
	main()
