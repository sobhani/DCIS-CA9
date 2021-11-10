import pathlib

from sccnn_detection import mat_to_tf

if __name__ == '__main__':
    main_file_path = pathlib.Path(
        r'Z:\DCIS_Duke_Faranak\HDD\Foxp3ca9-Final\data\SCCNN-dataprep\detection')
    save_path = str(pathlib.Path(r'tfrecords\foxp2ca9-13'))

    train_filename = 'fTrain-SCCNN-detection-24102019'
    valid_filename = 'fValid-SCCNN-detection-24102019'
    # Higher step is faster to process but do not increase beyond 500000.
    # If Matlab throws can't serialize error reduce the step size.
    step = 10000

    mat_to_tf.run(main_input_path=main_file_path,
                  save_tf_path=save_path,
                  train_tf_filename=train_filename,
                  valid_tf_filename=valid_filename,
                  step=step)
