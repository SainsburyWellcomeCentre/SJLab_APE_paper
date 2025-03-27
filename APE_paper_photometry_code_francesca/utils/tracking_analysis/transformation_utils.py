import skimage
import skimage.transform
import numpy as np

# Code from Steve Lenzi

def get_box_coordinates_from_file(box_path):
    napari_fmt_coords = np.load(str(box_path))
    napari_fmt_coords = np.roll(napari_fmt_coords, 1, axis=1)
    new_box_coords = np.empty_like(napari_fmt_coords)
    new_box_coords[0] = napari_fmt_coords[1]
    new_box_coords[1] = napari_fmt_coords[0]
    new_box_coords[2] = napari_fmt_coords[2]
    new_box_coords[3] = napari_fmt_coords[3]
    return new_box_coords


def get_inverse_projective_transform(
    src=np.array([[0, 240], [0, 0], [600, 240], [600, 0]]),
    dest=np.array(
        [
            [27.08333156, 296.33332465],
            [77.49999672, 126.74999637],
            [628.41664697, 308.24999096],
            [607.33331426, 130.41666292],
        ]
    ),
    output_shape=(240, 600),
):
    """
    coordinates np.array([x1, y1], [x2, y2], [x3, y3], [x4, y4])
    x2y2-------------------x4y4
     |                       |
     |                       |
     |                       |
    x1y1-------------------x3y3
    :param output_shape:
    :param img:
    :param src: coordinates of the four corners of the 'source' i.e. the desired standard space
    :param dest: coordinates of the four corners of the actual arena
    :return: transformed image
    """
    p = skimage.transform.ProjectiveTransform()
    p.estimate(src, dest)
    return p

def projective_transform_tracks(
    Xin,
    Yin,
    observed_box_corner_coordinates,
    target_box_corner_coordinates=[[0, 240],
                                     [0, 0],
                                     [600, 240],
                                     [600, 0]]
):

    """
    To correct for camera angle artifacts, coordinates of the arena and its known real geometry are used to
    get a projective transform that can be applied to positional tracks or raw images.


        x2y2-------------x4y4        b--------------d
       /                  /          |              |
      /                  /           |              |
     /                  /            |              |
    x1y1-------------x3y3            a--------------c

    target_box_corner_coordinates = [a=[0, 240],
                                     b=[0, 0],
                                     c=[600, 240],
                                     d=[600, 0]]


    observed_box_corner_coordinates = [e=[ 19.59140375, 296.09195599],
                                       f=[ 64.97987079, 124.3052263],
                                       g=[628.02667714, 284.60120484],
                                       h=[596.42711148, 102.47279911]]
    :param Xin:
    :param Yin:
    :param observed_box_corner_coordinates:
    :param target_box_corner_coordinates:
    :return: x and y positional traces transformed into standard space
    """
    p = get_inverse_projective_transform(
        dest=observed_box_corner_coordinates,
        src=np.array(target_box_corner_coordinates),
    )
    new_track_x = []
    new_track_y = []
    for x, y in zip(Xin, Yin):
        inverse_mapped = p.inverse([x, y])[0]
        new_track_x.append(inverse_mapped[0])
        new_track_y.append(inverse_mapped[1])
    return new_track_x, new_track_y

