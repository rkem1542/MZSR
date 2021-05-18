import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def write_to_tfrecord(writer, label, LR):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[LR]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(label_path, LRpath, tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))
    LR_list = np.sort(np.asarray(glob.glob(LRpath)))
    offset=0

    fileNum=len(LR_list)

    labels=[]
    LRs=[]
    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        label=imread(label_list[n])
        LR=imread(LR_list[n])
        x, y, ch = label.shape
        x_,y_,ch_ = LR.shape
        for m in range(8):
            for i in range(0+offset,x-patch_h+1,stride):
                for j in range(0+offset,y-patch_w+1,stride):
                    patch_l = label[i:i + patch_h, j:j + patch_w]
                    #if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                    labels.append(augmentation(patch_l,m).tobytes())
            for i_ in range(0+offset,int(x_-patch_h+1),int(stride)):
                for j_ in range(0+offset,int(y_-patch_w+1),int(stride)):
                    patch_LR = LR[i_:i_ + int(patch_h), j_:j_ + int(patch_w)]
                    #if np.log(gradients(patch_LR.astype(np.float64)/255.)+1e-10) >= -6.0:
                    LRs.append(augmentation(patch_LR,m).tobytes())
         
    #np.random.shuffle(labels)
    #np.random.shuffle(LRs)
    print('Num of patches of labels:', len(labels))
    print('Num of patches of LRs:', len(LRs))
    print('Shape of label: [%d, %d, %d]' % (patch_h, patch_w, ch))
    print('Shape of LR:  [%d, %d, %d]' % (patch_h, patch_w, ch_))
    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(labels)):
        if i % 10000 == 0:
            print('[%d/%d] Processed' % ((i+1), len(labels)))
        write_to_tfrecord(writer, labels[i],LRs[i])
   
    writer.close()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (/data3/sjyang/DIV2K/DIV2K_train_HR_cropped)')
    parser.add_argument('--LRpath', dest='LRpath', help='Path to LR images (/data3/sjyang/DIV2K/DIV2K_train_LR_bicubic_cropped/X1')
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='train_SR_MZSR')
    
    options=parser.parse_args()

    labelpath=os.path.join(options.labelpath, '*.png')
    LRpath=os.path.join(options.LRpath, '*.png')
    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(labelpath, LRpath, tfrecord_file,96,96,96)
    print('Done')

