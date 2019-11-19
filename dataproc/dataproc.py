import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import cv2
import subprocess
from multiprocessing import Pool


def get_worm_data_files(root_dir):
    experiment_dirs = (os.path.join(root_dir, o) for o in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, o)))

    for ed in experiment_dirs:

        hdf5_files = sorted(list(glob.glob1(ed, "*.hdf5")))
        hdf5_count = len(hdf5_files)

        if hdf5_count == 2:

            features_file_found = False

            for f in hdf5_files:
                file_base_name = os.path.splitext(f)[0]
                # print(file_base_name[-9:])
                if file_base_name[-9:] == "_features":
                    features_file_found = True

            if features_file_found:
                # Return tuple(main_file, feature_file, containing_dir_name)
                containing_dir_name = os.path.basename(ed)
                yield (os.path.join(ed, hdf5_files[0]), os.path.join(ed, hdf5_files[1]), containing_dir_name)


def explore_worm_files(main_file, feature_file):
    exp_keys_main_file = ['experiment_info', 'full_data', 'mask', 'provenance_tracking', 'stage_log',
                          'stage_position_pix', 'timestamp', 'video_metadata', 'xml_info']

    with h5py.File(main_file, "r") as f:

        if list(f.keys()) != exp_keys_main_file:
            print("DISAGREEMENT")

        for k in f.keys():

            print("{}:".format(k))
            is_group = isinstance(f[k], h5py.Group)

            if is_group:
                print(list(f[k].keys()))

            else:  # ==> is dataset.
                dset = f[k]
                print(dset.shape)
                print(dset.dtype)
                print("ATTRS:\n{}".format(list(dset.attrs)))

            print("---")

        print("\n--------------------------------------------------------\n")

        mask = f["mask"]

        # save_video_ffmpeg(mask)

        for frame in mask[:100]:
            print(type(frame))
            print(frame.shape)
            plt.imshow(frame)
            plt.show()
            time.sleep(0.5)


def retrieve_example_frame(ith_file=35, ith_frame=100):
    mf, ff, _ = list(get_worm_data_files("/mnt/windows/openworm/data_01/"))[ith_file]
    with h5py.File(mf, "r") as f:
        example_frame = f["mask"][ith_frame]
    return example_frame


def rescale_frame(frame, size=(48, 64), interpolation=cv2.INTER_AREA):
    # print(frame)
    # print(type(frame))
    # print(frame.shape)
    # plt.imshow(frame, cmap="Greys_r")
    # plt.show()
    # NOTE: Below are interpolation options.
    # for interpolation in [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]:
    if frame.shape != (480, 640):
        print("WARNING: Frame size not (480, 640)!")
    resized = cv2.resize(frame, dsize=tuple(list(size)[::-1]), interpolation=interpolation)
    # print(frame)
    # print(type(resized))
    # print(resized.shape)
    # plt.imshow(resized, cmap="Greys_r")
    # plt.show()
    return resized


def rescale_mask_frames(mask, rescaled_size=(48, 64)):
    mask_rescaled = np.zeros((mask.shape[0], rescaled_size[0], rescaled_size[1]))
    for idx, frame in enumerate(mask):
        mask_rescaled[idx, :, :] = rescale_frame(frame, size=rescaled_size)
    return mask_rescaled


def rescale_data_mask(main_file, output_dir):
    filename = os.path.basename(main_file)
    file_base_name = os.path.splitext(filename)[0]
    with h5py.File(main_file, "r") as f:
        mask = f["mask"]
        mask_rescaled = rescale_mask_frames(mask)
    output_filepath = os.path.join(output_dir, file_base_name + " PROC.hdf5")
    with h5py.File(output_filepath, "w") as f:
        dset = f.create_dataset("mask", mask_rescaled.shape, dtype=mask_rescaled.dtype, data=mask_rescaled)
        # print(dset)


def save_video_ffmpeg(frames, limit=50, temp_folder="/mnt/windows/openworm/data_01_proc/temp"):
    # NOTE: Slow.

    for i in range(min(limit, len(frames))):
        plt.imshow(frames[i], cmap="Greys_r")
        plt.savefig(os.path.join(temp_folder, "file%02d.png" % i))

    os.chdir(temp_folder)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'])

    for file_name in glob.glob("*.png"):
        os.remove(file_name)


def animate_frames_matplotlib(frames, limit=50):
    anim_frames = []
    fig = plt.figure()
    for i in range(min(limit, len(frames))):
        anim_frames.append([plt.imshow(frames[i], cmap="Greys_r", animated=True)])

    anim = animation.ArtistAnimation(fig, anim_frames, interval=50, blit=True, repeat_delay=1000)
    # ani.save('movie.mp4')
    plt.show()


def get_worm_proc_data_files(root_dir):
    experiment_dirs = (os.path.join(root_dir, o) for o in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, o)))

    for ed in experiment_dirs:

        hdf5_files = sorted(list(glob.glob1(ed, "*.hdf5")))
        hdf5_count = len(hdf5_files)

        if hdf5_count == 1:

            # Return tuple(file, containing_dir_name)
            containing_dir_name = os.path.basename(ed)
            yield (os.path.join(ed, hdf5_files[0]), containing_dir_name)


def explore_worm_proc_files(file):
    exp_keys_main_file = ['mask']

    with h5py.File(file, "r") as f:

        if list(f.keys()) != exp_keys_main_file:
            print("DISAGREEMENT")

        mask = f["mask"]

        print(mask.shape)
        print(mask.dtype)

        print(mask[0].sum())
        print(mask[:20].sum())

        for frame in mask[:10]:
            print(type(frame))
            print(frame.shape)
            plt.imshow(frame)
            plt.show()
            time.sleep(0.5)


def rescale_data_mask_wrapper(args):
    mf, _, dirname = args
    output_dir = "/mnt/windows/openworm/data_01_proc/" + dirname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rescale_data_mask(mf, output_dir=output_dir)


if __name__ == "__main__":
    # for mf, ff, dirname in list(get_worm_data_files("/mnt/windows/openworm/data_01/"))[:1]:  # [:1]:
    #     explore_worm_files(mf, ff)

    # for mf, ff, dirname in list(get_worm_data_files("/mnt/windows/openworm/data_01/"))[:1]:  # [:1]:
    #     output_dir = "/mnt/windows/openworm/data_01_proc/" + dirname
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     rescale_data_masks(mf, output_dir=output_dir)

    with Pool(8) as p:
        p.map(rescale_data_mask_wrapper, get_worm_data_files("/mnt/windows/openworm/data_01/"))

    # for f, dirname in list(get_worm_proc_data_files("/mnt/windows/openworm/data_01_proc/"))[:1]:  # [:1]:
    #     explore_worm_proc_files(f)

    # example = retrieve_example_frame()
    # example = rescale_example_frame(example)

    pass
