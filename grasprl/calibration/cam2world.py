import itertools
from math import atan2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mujoco
import mujoco.renderer
def compute_camera_matrix(renderer, model,data,camera=None):
  """Returns the 3x4 camera matrix."""
  # If the camera is a 'free' camera, we get its position and orientation
  # from the scene data structure. It is a stereo camera, so we average over
  # the left and right channels. Note: we call `self.update()` in order to
  # ensure that the contents of `scene.camera` are correct.
  if camera  is not None:
    renderer.update_scene(data,camera=camera)
  else:
    renderer.update_scene(data)
  pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
  z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
  y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
  rot = np.vstack((np.cross(y, z), y, z))
  fov = model.vis.global_.fovy

  # Translation matrix (4x4).
  translation = np.eye(4)
  translation[0:3, 3] = -pos

  # Rotation matrix (4x4).
  rotation = np.eye(4)
  rotation[0:3, 0:3] = rot

  # Focal transformation matrix (3x4).
  focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * renderer.height / 2.0
  focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

  # Image matrix (3x3).
  image = np.eye(3)
  image[0, 2] = (renderer.width - 1) / 2.0
  image[1, 2] = (renderer.height - 1) / 2.0
  return image @ focal @ rotation @ translation

def worldtocamera(model,data,renderer):
  box_pos = data.geom_xpos[model.geom('box_2').id]
  box_mat = data.geom_xmat[model.geom('box_2').id].reshape(3, 3)
  box_size = model.geom_size[model.geom('box_2').id]
  offsets = np.array([-1, 1]) * box_size[:, None]
  xyz_local = np.stack(list(itertools.product(*offsets))).T
  xyz_global = box_pos[:, None] + box_mat @ xyz_local
  corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
  corners_homogeneous[:3, :] = xyz_global
    # Get the camera matrix.
  m = compute_camera_matrix(renderer, model,data,camera="top_down")
  xs, ys, s = m @ corners_homogeneous
  # x and y are in the pixel coordinate system.
  x = xs / s
  y = ys / s
  # Render the camera view and overlay the projected corner coordinates.
  pixels = renderer.render()
  fig, ax = plt.subplots(1, 1)

  ax.imshow(pixels)
  print("x:{},y:{}".format(x,y))
  ax.plot(x, y, '+', c='w')
  ax.set_axis_off()
  plt.show()



def pixel_to_world(pixel_x,pixel_y,image_width,image_height,model):
    camera_depth = 0.61
    fovy = model.vis.global_.fovy
    camera_x = (pixel_x - image_width / 2) / image_width * 2 * np.tan(fovy / 2)
    camera_y = (pixel_y - image_height / 2) / image_height * 2 * np.tan(fovy / 2)
    world_x = camera_x * camera_depth
    world_y = camera_y * camera_depth
    print("World Coordinates: ({}, {})".format(world_x, world_y))
def main():
  model = mujoco.MjModel.from_xml_path("grasprl/worlds/grasp.xml")
  data  = mujoco.MjData(model)
  renderer = mujoco.Renderer(model,height=220,width=220)
  mujoco.mj_forward(model, data)
  worldtocamera(model,data,renderer)
  pixel_to_world(pixel_x=157,pixel_y=113,image_width=220,image_height=220,model=model)
  # print(matrix)
  # print(matrix_free)
if __name__ == '__main__':
  main()