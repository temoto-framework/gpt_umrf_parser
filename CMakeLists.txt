cmake_minimum_required(VERSION 3.0.2)
project(gpt_umrf_parser)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  temoto_action_engine
)

catkin_package(
   CATKIN_DEPENDS rospy std_msgs temoto_action_engine
)

catkin_python_setup()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
    scripts/gpt_umrf_parser_node.py
    scripts/gpt_umrf_parser/gpt_umrf_parser_base.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
