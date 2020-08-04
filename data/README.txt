-------------------------------------------------------------------------------
* Princeton Adobe Photo Triage Benchmark
for "Automatic Triage for a Photo Series"
SIGGRAPH 2016
Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein

Please cite the following paper if you use this dataset. 
More information on the project websites <http://phototriage.cs.princeton.edu/>

Contact: Huiwen Chang at <huiwenc@cs.princeton.edu>
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
* Contents

This package includes the following folders and files:

- train_pairlist.txt lists the pairs in all training photo series. The format is "#SERIES_ID #PHOTO1_IND #PHOTO2_IND #PREFERENCE_RATIO_of_PHOTO1_OVER_PHOTO2 #RANK_of_PHOTO1 #RANK_of_PHOTO2" 

- val_pairlist.txt list all the pairs in all validation photo series. If you want to test the performance of learning the human preferences offline, then save the result of your predictor into a textfile and run test.m. Your result could be either binary or float for the preferene of PHOTO1 over PHOTO2 for each pair.

- train_val_series.mat lists more information about the testing photo series, such as the Bradley-Terry scores modelled from human preferences.

- train_val_imgs/ includes all the images which are resized in 800x800 with its aspect ratio preserved. The format is #SERIES_ID(%06d)-#PHOTO_IND(%02d).JPG".

-------------------------------------------------------------------------------

