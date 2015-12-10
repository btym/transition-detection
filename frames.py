from PIL import Image
import numpy as np
import subprocess as sp

empty_frame = lambda H, W: np.array([[0]*H]*W)

def frames(path, W, H):
    """
    :param path: Path to an ffmpeg-readable file
    :param W: Resize width.
    :param H: Resize height.
    :return: List of 2D numpy arrays representing 0-255 grayscale pixel values.
    """
    frame_list = []
    command = [ 'ffmpeg',
            '-i', path,
            '-vf', 'scale=%i:%i' % (W, H),
            '-f', 'image2pipe',
            '-pix_fmt', 'gray',
            '-loglevel', 'panic',
            '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    while True:
        try:
            raw_image = pipe.stdout.read(W*H)
            frame_list.append(np.reshape(np.fromstring(raw_image, dtype='uint8'), (W, H)))
        except ValueError:
            break
    return frame_list


def get_transitions(path, W=32, H=32,
                    hotspot_history_size = 0.5 * 24,
                    hotspot_change_threshold = 0.25 * 255,
                    background_threshold = 0.8 * 255,
                    pixel_change_threshold = 0.3 * 255,
                    scene_change_threshold = 0.55,
                    total_background_threshold = 0.05,
                    image_output_dir=None):
    """

    :param path: Path to an ffmpeg-readable file.
    :param W: Width of the ffmpeg output to work on. Higher values = highers accuracy and lower performance.
    :param H: Height of the ffmpeg output to work on. Higher values = highers accuracy and lower performance.
    :param hotspot_history_size: Amount of hostpot changes to hold on to.
    :param hotspot_change_threshold: Amount a pixel has to change to increase its hotspot value.
    :param background_threshold: Hotspot value needed by a pixel to not be counted as background.
    :param pixel_change_threshold: Amount a background pixel needs to change to count towards a possible scene change.
    :param scene_change_threshold:  Percent of changed background pixels needed to count as a scene change
    :param total_background_threshold: Scene change will be forced if background pixels / total pixels falls under this.
    :param image_output_dir: If set, per-frame visualizations will be saved here.
    :return: List of transition frame numbers.
    """

    transitions = []

    hotspots = empty_frame(H, W)
    previous_frame = None
    hotspot_change_history = []


    for frame_num, this_frame in enumerate(frames(path, W, H)):
        transition_frame = False

        if previous_frame is None:  # First frame
            previous_frame = this_frame
            transitions.append(frame_num)
            continue

        change_this_frame = np.abs(this_frame - previous_frame)
        mean_change = np.mean(change_this_frame)
        hotspot_changes = empty_frame(H, W)

        for (x, y), change in np.ndenumerate(change_this_frame):
            if change > mean_change and change >= hotspot_change_threshold:
                hotspot_changes[x][y] += 50

        hotspots += hotspot_changes
        hotspot_change_history.append(hotspot_changes)  # Hold on to hotspot changes so they can be reversed later

        if len(hotspot_change_history) == hotspot_history_size+1:
            hotspots -= hotspot_change_history[0]  # Dispose of old hotspots
            hotspot_change_history.pop(0)

        background_pixels = np.vectorize(lambda pixel: 0 if pixel < background_threshold else 255)(hotspots)
        changed_background_pixel_count = 0
        total_background_pixel_count = 0

        for (x, y), pixel_value in np.ndenumerate(background_pixels):  # Determine which pixels are background
            if pixel_value == 0:
                total_background_pixel_count += 1
                if change_this_frame[x][y] >= pixel_change_threshold:
                    changed_background_pixel_count += 1

        if float(total_background_pixel_count)/float(W*H) < total_background_threshold or\
                total_background_pixel_count > 0 and\
                float(changed_background_pixel_count)/float(total_background_pixel_count) >= scene_change_threshold:
            transitions.append(frame_num)
            hotspots = empty_frame(H, W)
            transition_frame = True

        if image_output_dir is not None:
            dir_fmt = ('%s/%08d_' % (image_output_dir, frame_num))+'%s.jpg'
            Image.fromarray(this_frame).convert('RGB').save(dir_fmt % 'orig')
            Image.fromarray(background_pixels).convert('RGB').save(dir_fmt % 'bg')
            Image.fromarray(change_this_frame).convert('RGB').save(dir_fmt % 'change')
            Image.fromarray(np.array([[0 if transition_frame else 255]*H]*W)).convert('RGB').save(dir_fmt % 'trns')

        previous_frame = this_frame

    return transitions
