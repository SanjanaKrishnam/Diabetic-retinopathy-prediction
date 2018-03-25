import os
import pandas as pd

def get_lst_images(file_path):
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':
    trainLabels = pd.read_csv('/Users/Sanjana/Desktop/DR/trainLabels_master.csv')

    lst_imgs = get_lst_images('/Users/Sanjana/Desktop/DR/train-resized-256/')

    new_trainLabels = pd.DataFrame({'image': lst_imgs})
    new_trainLabels['image2'] = new_trainLabels.image

    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')


    new_trainLabels.columns = ['train_image_name', 'image']

    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis=1, inplace=True)
    trainLabels = trainLabels.dropna()
    print(trainLabels.shape)

    print("Writing CSV")
    trainLabels.to_csv('/Users/Sanjana/Desktop/DR/trainLabels_master_256_v2.csv', index=False, header=True)