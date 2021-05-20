from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
# Noise Level
parser.add_argument('--noise_std', type=float, dest='noise_std', default=0.0)
# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default='/data3/sjyang/SIDD/SIDD_test_LQ/')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='/data3/sjyang/SIDD/SIDD_test_HR/')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='')
parser.add_argument('--savepath', type=str, dest='savepath', default='/data3/sjyang/MZSR/result/SIDD/')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=10)

# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--train', dest='is_train', default=False, action='store_true')

args= parser.parse_args()

#Transfer Learning From Pre-trained model.
IS_TRANSFER = True
TRANS_MODEL = 'Large-Scale_Training/SR/Model0/model-800000'

# Dataset Options
HEIGHT=96
WIDTH=96
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[1.0]

META_ITER=100000
META_BATCH_SIZE=5
META_LR=1e-4

TASK_ITER=5
TASK_BATCH_SIZE=2
TASK_LR=1e-2

# Loading tfrecord and saving paths
TFRECORD_PATH='/data3/sjyang/MZSR/tfrecord/MZSR_denoising_SIDD_metalearn.tfrecord'
CHECKPOINT_DIR='/data3/sjyang/MZSR/MZSR_-interp_checkpoint/SR_MetaLearn_test_SIDD'
