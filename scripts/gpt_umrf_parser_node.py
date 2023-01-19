#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from temoto_action_engine.msg import BroadcastStartUmrfGraph
import os
import argparse
import gpt_umrf_parser.gpt_umrf_parser_base as gpt

class GptUmrfNode:
		def __init__(self, umrf_parser):
				self.umrf_parser = umrf_parser

				rospy.init_node('gpt_umrf_parser')
				self.umrf_pub = rospy.Publisher('/broadcast_start_umrf_graph', BroadcastStartUmrfGraph, queue_size=1)
				rospy.Subscriber("command", String, self.command_cb)

		def command_cb(self, msg):
				rospy.loginfo("Got command: %s\n", msg.data)
				umrf_json_str, query_metadata = self.umrf_parser.text_to_umrf(msg.data)
				umrf_json_str_pretty = gpt.to_pretty_json_str(umrf_json_str)
				rospy.loginfo("GPT returned:\n%s\n", umrf_json_str_pretty)

				umrf_graph_msg = BroadcastStartUmrfGraph()
				umrf_graph_msg.umrf_graph_name = msg.data
				umrf_graph_msg.name_match_required = False
				umrf_graph_msg.targets.append("everybody")
				umrf_graph_msg.umrf_graph_json = umrf_json_str
				self.umrf_pub.publish(umrf_graph_msg)

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('-ue', '--umrf-examples', required=True, type=str, help='Path for the generated C++ source files')
		args, unknown = parser.parse_known_args()

		examples = gpt.read_sample_jsons(args.umrf_examples)
		gpt_config = gpt.GptConfig(
				api_key = os.getenv('GPT_API_KEY'),
				engine = "text-davinci-003",
				logprobs = None,
				max_tokens = 1024
		)
		umrf_parser = gpt.GptUmrfParser(gpt_config, examples)

		try:
				gpt_umrf_node = GptUmrfNode(umrf_parser)
				rospy.loginfo("GPT UMRF parser up and running")
				rospy.spin()
		except rospy.ROSInterruptException:
				pass