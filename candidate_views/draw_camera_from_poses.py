import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import open3d as o3d
import matplotlib.tri as mtri
from matplotlib import cm
from scipy.spatial.transform import Rotation as R


def get_camera_wireframe(scale: float = 0.3):  # pragma: no cover
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    scale = 0.1
    a = scale * np.asarray([-2, 1.5, -4])
    up1 = scale * np.asarray([0, 1.5, -4])
    up2 = scale * np.asarray([0, 2, -4])
    b = scale * np.asarray([2, 1.5, -4])
    c = scale * np.asarray([-2, -1.5, -4])
    d = scale * np.asarray([2, -1.5, -4])
    C = np.zeros(3)
    F = scale * np.asarray([0, 0, -3])
    camera_points = np.asarray([a, b, d, c, a, C, b, d, C, c, C], dtype='float')
    camera_points = camera_points/2
    camera_points = camera_points.T
    camera_points = np.row_stack((camera_points, np.asarray([1 for i in range(np.shape(camera_points)[1])])))
    return camera_points


def draw_camera(camera_pose, ax):
    cam_wires_canonical = get_camera_wireframe()
    rlt = np.matmul(camera_pose, cam_wires_canonical)
    rlt = rlt[0:3]
    rlt = rlt.T
    for k in range(np.shape(rlt)[0] - 1):
        ax.plot([rlt[k][0], rlt[k + 1][0]],
                [rlt[k][1], rlt[k + 1][1]],
                [rlt[k][2], rlt[k + 1][2]],
                color="#BF1E2E", linewidth=0.8)


def cal_pose(angle_x, angle_y, angle_z):
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # print("R: {}".format(R))
    # print("R[:, 2]: {}".format(R[:, 2]))
    # print("np.expand_dims(R[:, 2], 1): {}".format(np.expand_dims(R[:, 2], 1)))

    # Set camera pointing to the origin(0, 1, 0) and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1) * 1
    # print("t: {}".format(t))

    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    # print("pose: {}".format(pose))

    return t, pose

def get_camera_pose(camera_poses, camera_id):
    pose_from_file = camera_poses[camera_id]
    position = [pose_from_file[0], pose_from_file[1], pose_from_file[2]]
    quaternion = [pose_from_file[3], pose_from_file[4], pose_from_file[5], pose_from_file[6]]
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    transformation = [[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], position[0]],
                        [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], position[1]],
                        [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], position[2]],
                        [0, 0, 0, 1]]
    return transformation

####################################
# main function
####################################
if __name__ == '__main__':

    Axes3D = Axes3D  # pycharm auto import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # draw mesh using open3d & matplot
    mesh = o3d.io.read_triangle_mesh('./model.obj')
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    vertices = np.asarray(mesh.vertices)
    vertices_t = vertices.T
    triang = mtri.Triangulation(vertices_t[0], vertices_t[1], np.asarray(mesh.triangles))
    ax.plot_trisurf(triang, vertices_t[2], alpha=0.3, linewidth=0, antialiased=False, color="#0D4C6D")
    # ax.view_init(elev=22, azim=-162, vertical_axis="y")
    ax.view_init(elev=16, azim=142, vertical_axis="y")
    # make the panes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


    # cnt = 0
    # for i in range(-90, 45, 30):
    #     angle_x = i * np.pi / 180
    #     for j in range(0, 360, 45):
    #         # discard 7 duplicate points
    #         cnt += 1
    #         if cnt < 8:
    #             continue
    #         angle_y = j * np.pi / 180
    #         t, pose = cal_pose(angle_x, angle_y, 0)
    #         draw_camera(pose, ax)

    camera_poses = np.loadtxt("./poses.txt")
    pose_size = camera_poses.shape[0]
    for i in range(pose_size):
        pose =  get_camera_pose(camera_poses, i)
        draw_camera(pose, ax)

    ax.set_xlabel("x (m)", fontsize='large')
    ax.set_ylabel("z (m)", fontsize='large')
    ax.set_zlabel("y (m)", fontsize='large')
    plt.show()


####################################
# reference function from pytorch3d
####################################
def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe().cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
    return fig
