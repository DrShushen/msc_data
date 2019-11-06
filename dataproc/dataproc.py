import os
import glob


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
                yield (hdf5_files[0], hdf5_files[1])


if __name__ == "__main__":
    pass
    # print(list(get_worm_data_files("/mnt/windows/openworm_db_filters/")))
    # print(len(list(get_worm_data_files("/mnt/windows/openworm_db_filters/"))))
