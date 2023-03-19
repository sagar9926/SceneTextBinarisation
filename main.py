from evaluation_fn import *
from extraction import *
from kapur_sahoo_wong import *
from preprocessing import *
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    image_path = os.path.join(os.path.join(os.getcwd(), 'SVT_Dataset'), 'img')
    label_path1 = os.path.join(os.path.join(os.getcwd(), 'SVT_Dataset'), 'test.xml')
    label_path2 = os.path.join(os.path.join(os.getcwd(), 'SVT_Dataset'), 'train.xml')

    img_name_1, label_1 = extract_labels(label_path1)
    img_name_2, label_2 = extract_labels(label_path2)
    img_name = np.concatenate((img_name_1, img_name_2))
    label = np.concatenate((label_1, label_2))

    imgs = load_img(image_path)
    gray_images = convert_to_gray(imgs)

    hist_eq_gray_imgs = equalize_hist(gray_images)
    bin_imgs = []
    bin_imgs_enhs = []
    for img in gray_images:
        th = get_kapur_threshold(img)
        bin_imgs.append(cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1])

    for img in hist_eq_gray_imgs:
        th = get_kapur_threshold(cv2.equalizeHist(img))
        bin_imgs_enhs.append(cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1])

    save_images(bin_imgs, img_name, 'kapur')
    save_images(bin_imgs_enhs, img_name, 'kapur-enhs')

    data = np.vstack((img_name, label))

    df = pd.DataFrame(data.T, columns=['Image_Name', 'Text_Present'])
    df.to_csv(os.path.join(os.getcwd(), 'label_text.csv'), index=False)

    kapur_key = extract_text_from_image('kapur')
    kapur_enhn_key = extract_text_from_image('kapur-enhs')

    df1 = pd.DataFrame(np.array(kapur_key), columns=['Text_present'])
    df2 = pd.DataFrame(np.array(kapur_enhn_key), columns=['Text_present'])

    df1.to_csv(os.path.join(os.getcwd(), 'Kapur_text.csv'), index=False)
    df2.to_csv(os.path.join(os.getcwd(), 'Kapur_enhance_text.csv'), index=False)

    score1 = evaluate(df['Text_Present'], df1)
    score2 = evaluate(df['Text_Present'], df2)
    print(pd.merge(score1, score2).describe())
