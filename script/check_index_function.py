import numpy as np
class dummy():
    def __init__(self, gridsize=(3,3), resolution=1):
        self.gridsize = [3,3]
        self.resolution = 0.5

     
    def in_which_cell(self, pose):
            # pose = [-pose[1], pose[0]]

            if pose[0] < self.gridsize[1]*self.resolution - 0.5*self.resolution and pose[0] > -0.5*self.resolution \
                and pose[1] > -0.5*self.gridsize[0]*self.resolution and pose[1] < 0.5*self.gridsize[0]*self.resolution:

                # pose[1] = max(0,pose[1])
                
                # y = min(((self.gridsize[1])*self.resolution - pose[1]) // self.resolution, 2)
                y = ((self.gridsize[1])*self.resolution/2 - pose[1]) // self.resolution

                x = ((-pose[0] + (self.gridsize[1]+0.5)*self.resolution) // self.resolution) -1
                # print([x, y]) # (1,2) -> (1,1) -> (0,1)
                if (x<0 or y<0):
                    return None
                return [int(x), int(y)]
            else:
                return None

if __name__ == "__main__":
    dummy = dummy()
    new_pose = []
    cell_size = (dummy.gridsize[0]+2)*dummy.resolution 
    x = np.arange(-1,cell_size, dummy.resolution/2)
    y = np.arange(-cell_size/2,cell_size/2, dummy.resolution/2)
    for i in x:
         for j in y:
              if (dummy.in_which_cell([i,j])):
                new_pose.append([[i,j],[dummy.in_which_cell([i,j])]])
    print(new_pose)
   