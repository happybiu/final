import matplotlib.pyplot as plt
import numpy as np
import cv2
import face_recognition


def get_img_pairs_list(pairs_txt_path, img_path):
    """ 指定图片组合及其所在文件，返回各图片对的绝对路径
        Args:
            pairs_txt_path：图片pairs文件，里面是6000对图片名字的组合
            img_path：图片所在文件夹
        return:
            img_pairs_list：深度为2的list，每一个二级list存放的是一对图片的绝对路径
    """
    file = open(pairs_txt_path)
    img_pairs_list, labels = [], []
    # count = 0
    while 1:
        # count = count + 1
        img_pairs = []
        line = file.readline().replace('\n','')
        # if count > 601:
        #     break
        if line == '':
            break
        line_list = line.split('\t')
        if len(line_list) == 3:
            # 图片路径示例：
            # 'C:\Users\thinkpad1\Desktop\image_set\lfw_funneled\Tina_Fey\Tina_Fey_0001.jpg'
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[1])[-4:]+'.jpg')
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[2])[-4:]+'.jpg')
            labels.append(1)
        elif len(line_list) == 4:
            img_pairs.append(img_path+'\\'+line_list[0]+'\\'+line_list[0]+'_'+('000'+line_list[1])[-4:]+'.jpg')
            img_pairs.append(img_path+'\\'+line_list[2]+'\\'+line_list[2]+'_'+('000'+line_list[3])[-4:]+'.jpg')
            labels.append(0)
        else:
            continue
        
        img_pairs_list.append(img_pairs)
    return img_pairs_list, labels


def roc(dist,labels):
    TP_list, TN_list, FP_list, FN_list,TPR,FPR = [],[],[],[],[],[]
    for t in range(180):
        threh = 0.1+t*0.01
 
        TP,TN,FP,FN = 0,0,0,0
        for i in range(len(dist)):
            if labels[i]==1 and dist[i]!=-1:
                if dist[i]<threh:
                    TP += 1
                else:
                    FN += 1
            elif labels[i]==0 and dist[i]!=-1:
                if dist[i]>=threh:
                    TN += 1
                else:
                    FP += 1
        TP_list.append(TP)
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
    return TP_list,TN_list,FP_list,FN_list,TPR,FPR


def get_imgs_dist(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    # small_img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25)
    # small_img2 = cv2.resize(img2, (0, 0), fx=0.25, fy=0.25)
    rgb_small_img1 = img1[:, :, ::-1]
    rgb_small_img2 = img2[:, :, ::-1]
    img1_face_location = face_recognition.face_locations(rgb_small_img1)
    if (len(img1_face_location) != 1):
        return -1
    img2_face_location = face_recognition.face_locations(rgb_small_img2)
    if (len(img2_face_location) != 1):
        return -1
    img1_face_encoding = face_recognition.face_encodings(rgb_small_img1, img1_face_location)
    img2_face_encoding = face_recognition.face_encodings(rgb_small_img2, img2_face_location)
    
    return face_recognition.face_distance(img1_face_encoding, img2_face_encoding[0])[0]

def cmc(predict_label, test_y):
    # CMC曲线
    # 需要提供predict_label和test_y这两个变量
    # 需要安装numpy和matplotlib这两个包
    test_cmc = []  #保存accuracy，记录rank1到rank48的准确率
    sort_index = np.argsort(predict_label,axis=1)#predict_label为模型预测得到的匹配分数矩阵；降序排序，返回匹配分数值从大到小的索引值

    actual_index = np.argmax(test_y,1) #test_y为测试样本的真实标签矩阵；返回一列真实标签相对应的最大值的索引值
    predict_index = np.argmax(predict_label,1)#返回一列预测标签相对应的最大值的索引值

    temp = np.cast['float32'](np.equal(actual_index,predict_index)) #一列相似值，1代表相同，0代表不同
    test_cmc.append(np.mean(temp))#rank1
    #rank2到rank48
    for i in range(sort_index.shape[1]-1):
        for j in range(len(temp)):
            if temp[j]==0:
                predict_index[j] = sort_index[j][i+1]
        temp = np.cast['float32'](np.equal(actual_index,predict_index))
        test_cmc.append(np.mean(temp)) 
    #创建绘图对象  
    plt.figure()
    x = np.arange(0,sort_index.shape[1])  
    plt.plot(x,test_cmc,color="red",linewidth=2)
    plt.xlabel("Rank")  
    plt.ylabel("Matching Rate") 
    plt.legend() 
    plt.title("CMC Curve")
    plt.savefig('D:\MasterCourse\CV\homework\\final\curve_cmc_test.png')
    plt.show()


if __name__ == '__main__':
    pairs_txt_path = 'D:\MasterCourse\CV\homework\\final\LFW\pairs.txt'
    img_path = 'D:\MasterCourse\CV\homework\\final\LFW\lfw'

    img_pairs_list, labels = get_img_pairs_list(pairs_txt_path, img_path)
    dists = []
    count = 0
    for img_pair in img_pairs_list:
        dists.append(get_imgs_dist(img_pair[0], img_pair[1]))
        count = count + 1
        print(count)
        if count == 100:
            break

    TP_list,TN_list,FP_list,FN_list,TPR,FPR = roc(dists,labels)
    plt.plot(FPR,TPR,label='Roc')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('D:\MasterCourse\CV\homework\\final\curve_roc_test.png')
    plt.show()

    cmc(dists, labels)
    

