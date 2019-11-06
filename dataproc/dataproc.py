import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time


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
                # Return tuple(main_file, feature_file)
                yield (os.path.join(ed, hdf5_files[0]), os.path.join(ed, hdf5_files[1]))


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

        for frame in mask[:100]:
            print(type(frame))
            print(frame.shape)
            plt.imshow(frame)
            plt.show()
            time.sleep(0.5)


if __name__ == "__main__":

    for mf, ff in list(get_worm_data_files("/mnt/windows/openworm_db_filters/"))[:1]:  # [:1]:
        explore_worm_files(mf, ff)

    pass
