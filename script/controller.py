import rospy



class Controller():
    def __init__(self, p = 1, i = 1, d = 1):

        self.d_thres = 0.5
        
        
        pass

    def policy2traj(self, policy):
        '''
        Policy be like:
        [['r' 'r' 'r']
        ['r' 'r' 'r']
        ['r' 'r' 'r']]
        '''
        pass


    def get_next_pose(self, traj):
        
        pass


    def send_command(self, next_pose):
        pass

    def execute_policy(self, policy):

        pass