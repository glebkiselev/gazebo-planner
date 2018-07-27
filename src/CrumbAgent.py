import logging
import re

from gazeboplanner.gadapter.gazebo_tools import TurtleBot, Processer
from mapplanner.agent.agent_search import Agent

#ROS libs
import rospy
from std_srvs.srv import Empty

class GazeboAgent(Agent):
    def __init__(self):
        super().__init__()
        pass

    # Gazebo publisher
    def gazebo_visualization(self):
        print('at special visualization')
        plan_to_gazebo = []
        pr = Processer(10, 10, 20)
        for action in self.final_solution.strip().split(" && "):
            if '(' in action and ')' in action:
                action = re.split('[\(\)]', action)
                coords = action[1].split(', ')
                new_coords = pr.to_gazebo(float(coords[0]), float(coords[1]))
                plan_to_gazebo.append((action[0],new_coords[0],new_coords[1],action[2]))
        # init ROS node
        rospy.init_node('signs_server')
        empty_call = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        empty_call()
        tb = TurtleBot()
        prev_directions = []
        prev_poses = []

        for action in plan_to_gazebo:
            logging.info('Trying to implement action {0}'.format(action[0]))
            try:
                if not prev_directions and 'move' not in action[0]:
                    prev_direct = self.problem.initial_state["SpatialAgents-orientation"]
                    prev_directions.append(prev_direct)
                elif 'move' in action[0]:
                    prev_directions.append(action[3][1:])
                    prev_direct = prev_directions[-1]
                else:
                    prev_direct = prev_directions[-1]
                if not prev_poses:
                    prev_poses.append((0.0, 0.0))
                else:
                    prev_poses.append((action[1], action[2]))

                if  'move' in action[0]:
                    resp = tb.move(action[1], action[2], prev_direct, prev_poses[-1][0], prev_poses[-1][1])
                    if not resp:
                        raise Exception('Agent {0} can not implement a move action'.format(self.name))
                elif 'rotate' in action[0]:
                    prev_directions.append(action[3][1:])
                    resp = tb.rotate(action, prev_direct, prev_poses)
                    if not resp:
                        raise Exception('Agent {0} can not implement a rotate action'.format(self.name))
                elif 'pick-up' in action[0]:
                    resp = tb.pickup(action, prev_direct, 0.5)
                    if not resp:
                        raise Exception('Agent {0} can not implement a pick-up action'.format(self.name))
                elif 'put-down' in action[0]:
                    prev_direct = prev_directions[-1]
                    resp = tb.putdown(action, prev_direct, 0.5)
                    if not resp:
                        raise Exception('Agent {0} can not implement a put-down action'.format(self.name))
            except rospy.ROSInterruptException:
                raise Exception('Somebody Interrupt Agent! Call 911!')

