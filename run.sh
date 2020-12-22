#set -x
#QSUB_OPTIONS='-q lg-mem -l h_vmem=50G -M Gu.Qiangqiang@mayo.edu -m abe -V -cwd -j y -o /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/LOG'
img_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Split/train'
re_img_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Split'
patch_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/patches'
tf_dir='/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/tfrecord_c01/train'
is_reorganize='True',
is_split=args.is_split='True',
return_patches='True',
is_tfrecord_create='True'
patch_size=512
least_level=4
c0_train=1.0
c1_train=0.0
/research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/tf2/bin/python3 /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/pGANa_Preprocess_Clustering/main.py -n $is_reorganize -l $is_split -u $return_patches -e $is_tfrecord_create -i $img_dir -g $re_img_dir -a $patch_dir -t $tf_dir -p $patch_size -f $least_level -c $c0_train -r $c1_train
#qsub $QSUB_OPTIONS -b y /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Anaconda/conda_env/tf2/bin/python3 /research/bsi/projects/PI/tertiary/Hart_Steven_m087494/s211408.DigitalPathology/Quincy/Code/pGANa_Preprocess_Clustering/main.py -n $is_reorganize -l $is_split -u $return_patches -e $is_tfrecord_create -i $img_dir -g $re_img_dir -a $patch_dir -t $tf_dir -p $patch_size -f $least_level -c $c0_train -r $c1_train