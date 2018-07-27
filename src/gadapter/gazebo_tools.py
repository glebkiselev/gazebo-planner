#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from gazeboplanner.gadapter.agent_TRPO import TRPOAgent
from geometry_msgs.msg import Point

import time
from math import sqrt, acos, atan, radians, pow
import logging

import gym
import gym_crumb


from gazebo_msgs.srv import GetModelState


prev_directions = []
plan = []


class TurtleBot:

    def __init__(self):
        # Creates a node with name 'turtlebot_controller' and make sure it is a

        # Publisher which will publish to the topic '/turtle1/cmd_vel'.
        self.velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity',
                                                  Twist, queue_size=10)

        # A subscriber to the topic '/simulation/pose'. self.update_pose is called
        # when a message of type Point is received.
        # self.pose_subscriber = rospy.Subscriber('/simulation/pose',
        #                                         Point, self.update_pose)

        self.pose = Point()
        self.rate = rospy.Rate(10)

        # camera handler
        self.cam_sub = rospy.Subscriber("/camera/depth/points", PointCloud2, self.callback_kinect)
        self.middle = PointCloud2()

    def subscribe_pose(self, object, ob2 = ''):
        # A subscriber to the service '/simulation/pose'. self.update_pose is called
        # when a message of type Point is received.
        model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        return model_state(object, ob2)

    def callback_kinect(self, data):
        # pick a height
        height = int(data.height / 2)
        # pick x coords near front and center
        middle_x = int(data.width / 2)
        # examine point
        self.read_depth(middle_x, height, data)
        # do stuff with middle

    def read_depth(self, width, height, data):
        # read function
        if (height >= data.height) or (width >= data.width):
            return -1
        data_out = pc2.read_points(data, field_names=None, skip_nans=False, uvs=[[width, height]])
        int_data = next(data_out)
        #rospy.loginfo("int_data " + str(int_data))
        self.middle = int_data
        return int_data

    def update_pose(self, data):
        """Callback function which is called when a new message of type Pose is
     received by the subscriber."""
        self.pose = data
        self.pose.x = round(self.pose.x, 6)
        self.pose.y = round(self.pose.y, 6)

    def euclidean_distance(self, goal_pose):
        """Euclidean distance between current pose and the goal."""
        return sqrt(pow((goal_pose.x - self.pose.x), 2) +
                    pow((goal_pose.y - self.pose.y), 2))

    def euclidean_cur_distance(self, goal_pose, cur_pose_x, cur_pose_y):
        return sqrt(pow((goal_pose.x - cur_pose_x), 2) +
                    pow((goal_pose.y - cur_pose_y), 2))

    def move(self, pose_x, pose_y, direction, cur_pose_x = None, cur_pose_y = None):
        """Moves the turtle to the goal."""
        #print('poses are {0},{1}'.format(pose_x, pose_y))

        goal_pose = Point()

        goal_pose.x = pose_x
        goal_pose.y = pose_y
        goal_pose.z = 0

        resp = self.rotate_to_goal(goal_pose, direction, cur_pose_x, cur_pose_y)

        while not resp:
            time.sleep(1)

        vel_msg = Twist()

        vel_msg.linear.x = 0.4
        # Since we are moving just in x-axis
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        # TODO test that on 1 line or move diagonal/katiti
        while not rospy.is_shutdown():

            # Setting the current time for distance calculus
            t0 = rospy.Time.now().to_sec()
            current_distance = 0
            if not cur_pose_x:
                evcl = self.euclidean_distance(goal_pose)
            else:
                evcl = self.euclidean_cur_distance(goal_pose, cur_pose_x, cur_pose_y)
            # Loop to move the turtle in an specified distance
            while (current_distance < evcl):

                # Publish the velocity
                self.velocity_publisher.publish(vel_msg)
                # Takes actual time to velocity calculus
                t1 = rospy.Time.now().to_sec()
                # Calculates distancePoseStamped
                current_distance = vel_msg.linear.x * (t1 - t0)
            # After the loop, stops the robot
            vel_msg.linear.x = 0
            # Force the robot to stop
            self.velocity_publisher.publish(vel_msg)
            print('robot stoped')
            return 'yes'

        # If we press control + C, the node will stop.
        rospy.spin()

    def rotate_to_goal(self, goal_pose, direction, cur_pose_x=None, cur_pose_y=None):
        print('rotating to goal!')
        if not cur_pose_x and not cur_pose_y:
            cur_pose_x = self.pose.x
            cur_pose_y = self.pose.y

        if goal_pose.x >= cur_pose_x and goal_pose.y >= cur_pose_y:
            if sqrt(pow((goal_pose.x-cur_pose_x), 2)) > sqrt(pow((goal_pose.y - cur_pose_y), 2)):
                new_dir = 'right'
                clockwise = False
            else:
                new_dir = 'above'
                clockwise = False
        elif goal_pose.x <= cur_pose_x and goal_pose.y >= cur_pose_y:
            if sqrt(pow((cur_pose_x - goal_pose.x), 2)) > sqrt(pow((goal_pose.y - cur_pose_y), 2)):
                new_dir = 'left'
                clockwise = True
            else:
                new_dir = 'above'
                clockwise = False
        elif goal_pose.x >= cur_pose_x and goal_pose.y <= cur_pose_y:
            if sqrt(pow((goal_pose.x-cur_pose_x), 2)) > sqrt(pow((cur_pose_y - goal_pose.y), 2)):
                new_dir = 'right'
                clockwise = True
            else:
                new_dir = 'below'
                clockwise = False
        else:
            if sqrt(pow((goal_pose.x-cur_pose_x), 2)) > sqrt(pow((cur_pose_y - goal_pose.y), 2)):
                new_dir = 'left'
                clockwise = False
            else:
                new_dir = 'below'
                clockwise = True

        if abs(goal_pose.x) - abs(cur_pose_x) > abs(goal_pose.y) - abs(cur_pose_y):
            nearest = abs(abs(cur_pose_x) - abs(goal_pose.x))
            other = abs(abs(cur_pose_y) - abs(goal_pose.y))
        else:
            nearest = abs(abs(cur_pose_y) - abs(goal_pose.y))
            other = abs(abs(cur_pose_x) - abs(goal_pose.x))

        angle = atan(other / nearest)
        change_dir = False

        if new_dir != direction:
            print('new dir is {0}'.format(new_dir))
            resp = self.rotate(new_dir, direction)
            change_dir = True
        else:
            print('dir is the same')
            resp = 'yes'

        speed = 12.0

        if resp and change_dir:
            angular_speed = radians(speed)

            resp2 = self.rudder(angle, angular_speed, clockwise)

            if resp2:
                return resp2, angle, clockwise
        elif resp:
            return 'yes', angle, clockwise

    def get_angle(self, direct, prev_direct):
        circle = {}
        circle['above'] = 0
        circle["below"] = 180
        circle["left"] = -90
        circle["right"] = 90
        circle["above-left"] = -45
        circle["above-right"] = 45
        circle["below-left"] = -135
        circle["below-right"] = 135

        prev_angle = circle[prev_direct]
        dir_angle = circle[direct]
        if prev_angle < 0 and dir_angle > 0:
            ch_ang = abs(prev_angle) + dir_angle
        elif prev_angle < 0 and dir_angle < 0:
            ch_ang = dir_angle - prev_angle
        elif prev_angle > 0 and dir_angle < 0:
            ch_ang = prev_angle - dir_angle
        else:
            ch_ang = dir_angle - prev_angle
        return radians(ch_ang)

    def rudder(self, relative_angle, angular_speed=0.5, clockwise=True):


        vel_msg = Twist()
        # We wont use linear components
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        # Checking if our movement is CW or CCW
        if clockwise:
            vel_msg.angular.z = -abs(angular_speed)
        else:
            vel_msg.angular.z = abs(angular_speed)
        # Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        # t0 = self.time
        current_angle = 0

        while (current_angle < relative_angle):
            self.velocity_publisher.publish(vel_msg)
            t1 = rospy.Time.now().to_sec()
            current_angle = angular_speed * (t1 - t0)

        # Forcing our robot to stop
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        return 'yes'

    def tb2_fault(self, action, prev_poses):
        crumb_pose = self.subscribe_pose('crumb')
        AC_len = sqrt((crumb_pose.pose.position.x-prev_poses[-2][0])**2 + (crumb_pose.pose.position.y-prev_poses[-2][1])**2)
        BC_len = action[2] - crumb_pose.pose.position.y
        ACB = acos(abs(BC_len)/abs(AC_len))
        ACD = radians(90)
        return abs(ACD-ACB)

    def rotate(self, action, prev_direct, prev_poses = None):
        "rotate to new direction"
        direct = action[3][1:]
        print('rotating to {0}'.format(direct))
        PI = 3.1415926535897

        # degrees/sec
        speed = 12.0
        # direction
        print('new dir: {0}, prev dir: {1}'.format(direct, prev_direct))
        angle = self.get_angle(direct, prev_direct)

        if angle < 0:
            clockwise = False
        else:
            clockwise = True
        if prev_poses:
            additional_angle = self.tb2_fault(action, prev_poses)
        else:
            additional_angle = 0.0

        angle = abs(angle) + additional_angle*3
        # angle = abs(angle)

        # Converting from angles to radians
        angular_speed = speed * 2 * PI / 360


        return self.rudder(angle, angular_speed, clockwise)


    def pickup(self, act_form, prev_direct, table_radius):

        # cp_x = act_form[1]
        # cp_y = act_form[2]
        # modified = round(cp_y+ cp_x - 0.4, 2)

        crumb_pose = self.subscribe_pose('crumb')
        unit_box_3_pose = self.subscribe_pose('unit_box_3')


        resp = self.move(unit_box_3_pose.pose.position.x - (table_radius/2), unit_box_3_pose.pose.position.y - (table_radius/2), prev_direct, crumb_pose.pose.position.x, crumb_pose.pose.position.y)

        new_env = gym.make("crumb-synthetic-v0")
        TRPOagent = TRPOAgent(new_env)
        logging.info('env was made')
        TRPOagent.net.Loadmodel()
        env = gym.make("crumb-pick-v0")
        TRPOagent.grasp(env)
        TRPOagent.pick(env)

        return 'yes'

    def putdown(self, act_form, prev_direct, table_radius):
        crumb_pose = self.subscribe_pose('crumb')
        unit_box_1_pose = self.subscribe_pose('unit_box_1')

        resp = self.move(unit_box_1_pose.pose.position.x, unit_box_1_pose.pose.position.y, prev_direct,
                         crumb_pose.pose.position.x, crumb_pose.pose.position.y)

        new_env = gym.make("crumb-synthetic-v0")
        TRPOagent = TRPOAgent(new_env)
        TRPOagent.net.Loadmodel()
        env = gym.make("crumb-pick-v0")
        TRPOagent.putdown(env)

        return 'yes'

class Processer:
    # gazebo coords and koef of sign blow
    def __init__(self, gazX, gazY, koef):
        self.mapX = gazX*koef #200
        self.mapY = gazY*koef #200
        self.koef = koef #20

    def to_gazebo(self, x, y): #
        if x < self.mapX // 2 and y <self.mapY // 2:
            gaz_y = (self.mapY // 2 - y) / self.koef * 2
            gaz_x = (self.mapX // 2 - x) / self.koef * 2 * (-1)    # because in 2 quadro
        elif x > self.mapX // 2 and y <self.mapY // 2: # 1 quadro
            gaz_y = (self.mapY // 2 - y) / self.koef * 2
            gaz_x = (x - self.mapX // 2) / self.koef * 2
        elif x > self.mapX // 2 and y > self.mapY // 2: # 4 quadro
            gaz_y = (y - self.mapY // 2) / self.koef * 2 * (-1)
            gaz_x = (x - self.mapX // 2) / self.koef * 2
        else: # 3 quadro
            gaz_y = (y - self.mapY // 2) / self.koef * 2 * (-1)
            gaz_x = (self.mapX // 2 - x) / self.koef * 2 * (-1)
            if gaz_x == 0.0:
                gaz_x *= -1
            if gaz_y == 0.0:
                gaz_y *= -1

        return gaz_x, gaz_y

    def to_signs(self, x, y): # 2.5 2
        if x > 0.0 and y > 0.0: # 1 quadro
            sig_y = self.mapY //2 - y*self.koef//2
            sig_x = self.mapX //2 + x*self.koef//2
        elif x > 0.0 and y < 0.0: # 4 quadro
            sig_y = self.mapY //2 + abs(y)*self.koef//2
            sig_x = self.mapX //2 + x*self.koef//2
        elif x<0.0 and y <0.0: # 3 quadro
            sig_y = self.mapY //2 + abs(y)*self.koef//2
            sig_x = self.mapX //2 - abs(x)*self.koef//2
        else: # 2 quadro
            sig_y = self.mapY //2 - y*self.koef//2
            sig_x = self.mapX //2 - abs(x)*self.koef//2
        return sig_x, sig_y

