import torch.utils.data as data
import os
import os.path
import numpy as np
import csv
import open3d as o3d
import copy
import sys
sys.path.append('./')


def pc_normalize(data):
    centroid = np.mean(data, axis=0)
    pc = data - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_np(file):
    data = np.loadtxt(file, delimiter=',', dtype=float, encoding='utf-8')
    return data

def o3d_random_rotate(xyz, axi):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    random_angle = np.random.uniform() * 2 * np.pi

    if axi == 'x':
        rotate_pcd = copy.deepcopy(pcd)
        random_angle = pcd.get_rotation_matrix_from_xyz((random_angle, 0, 0))
        rotate_pcd.rotate(random_angle)
    elif axi == 'y':
        rotate_pcd = copy.deepcopy(pcd)
        random_angle = pcd.get_rotation_matrix_from_xyz((0, random_angle, 0))
        rotate_pcd.rotate(random_angle)
    else:
        rotate_pcd = copy.deepcopy(pcd)
        random_angle = pcd.get_rotation_matrix_from_xyz((0, 0, random_angle))
        rotate_pcd.rotate(random_angle)   

    xyz_new = np.asarray(rotate_pcd.points)
    return xyz_new

def data_aug(path, index, num_aug, start):
    sample_path = os.path.join(path, 'sample_normalized')
    truth_path = os.path.join(path, 'truth_normalized')
    sample_file = os.path.join(sample_path, index)
    truth_file = os.path.join(truth_path, index)
    # print(get_np(truth_file))
    data = np.concatenate((get_np(sample_file), get_np(truth_file)))
    # print(data.shape)
    sample_aug_path = os.path.join(path, 'sample_aug')
    truth_aug_path = os.path.join(path, 'truth_aug')

    if os.path.exists(sample_aug_path) == False:
        os.makedirs(sample_aug_path)
    if os.path.exists(truth_aug_path) == False:
        os.makedirs(truth_aug_path)

    ind = start
    for i in range(num_aug):
        for axi in ['x', 'y', 'z']:
            rotated = o3d_random_rotate(data, axi)
            # print(rotated.shape)
            rotated = pc_normalize(rotated)
            # print(rotated.shape)
            split = np.array_split(rotated, indices_or_sections=[len(get_np(sample_file)), len(rotated)], axis=0)
            # print('---------数据分割结果---------', '\n', 'sample长度：%s \n truth长度%s' % (str(split[0].shape), str(split[1].shape)))

            name_index = str(ind).zfill(5) + '.csv'
            print('当前增强数据为：%s, 增强后文件：%s' % (index, name_index))

            with open(os.path.join(sample_aug_path, name_index), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for n in split[0]:
                    writer.writerow(n)
                f.close()
            
            with open(os.path.join(truth_aug_path, name_index), 'w', newline='', encoding='utf-8') as g:
                writer1 = csv.writer(g)
                # print(split[1])
                for m in split[1]:
                    writer1.writerow(m)
                g.close()
            ind += 1    
    return ind

if __name__ == '__main__':
    path = r'H:\irregular construct\dataset'
    file_list = os.listdir(os.path.join(path, 'sample'))

    ind = 0
    for i in file_list:
        num_aug = 8
        ind = data_aug(path, i, 12, ind)
        print('----------------', '\n', '数据增强已完成，增强数%s， 增强数据为：%s' %(str(num_aug), i))