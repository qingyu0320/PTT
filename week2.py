import os
import jieba
import pandas as pd
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
import imageio.v2
import matplotlib.pylab as plt

os.chdir(r"C:/Users/thinkpad/Desktop/codework/python/PTT/week2")

#�����ض���Ƶ���Ƶ�ʺ���
def word_search(value, requested):
    result = set()
    #(-1, 0, 1) mains '(<, =, >)'
    for item in words_count.items():
        if requested  ==  '-1' and item[1] < value:
            result.add(item[0])
        elif requested == '0' and item[1] == value:
            result.add(item[0])
        elif requested == '1' and item[1] > value:
            result.add(item[0])
    return list(result)

#������ɸѡ����
def word_screen(start, end):
    #end=-1 mains '>=start' start=-1 mains '<=end'
    if end == -1:
        result = word_search(start-1, requested = '1')
    elif start == -1:
        result = word_search(end+1, requested = '-1')
    else:
        result_1 = word_search(start-1, requested = '1')
        result_2 = word_search(end+1, requested = '-1')
        result = (result_1 & result_2)
    return list(result)

#�����������ɺ���
def word_vector(word):
    array = np.zeros(len(words_feature), dtype = np.int)
    for i in word:
        if i in words_feature:
            array[words_feature.index(i)] = 1
    return list(array)

#ŷ����þ������ƶȺ���
def euclidean_similarity(indexes1, indexes2):
    similarity_list = []
    vector1 = np.array([words_vector[i] for i in indexes1])
    vector2 = np.array([words_vector[j] for j in indexes2])
    for i in range(len(indexes2)):
        distance = np.sqrt(np.sum(np.square(vector1 - vector2[i]), axis = 1))
        similarity = np.round(1 / (1 + distance), 4)
        similarity_list.append(list(similarity))
    return similarity_list

#�������ƶȺ���
def cosine_similarity(indexes1, indexes2):
    similarity_list = []
    vector1 = np.array([words_vector[i] for i in indexes1])
    vector2 = np.array([words_vector[j] for j in indexes2])
    for i in range(len(indexes2)):
        product_ab = np.dot(vector1, vector2[i])
        modulus_a, modulus_b = np.linalg.norm(vector1, axis = 1), np.linalg.norm(vector2[i])
        similarity = np.round(product_ab / (modulus_a * modulus_b), 4)
        similarity[np.isnan(similarity)] = 0
        similarity_list.append(list(similarity))
    return similarity_list

#���ӻ����ƺ���
    #top_num:����ӻ��ĸ�Ƶ������
def draw_word_cloud(top_num, all_words):
    #���ֵ������Ƭ����
    top_words = {}
    for i in range(top_num):
        key = all_words[i][0]
        value = all_words[i][1]
        top_words[key] = value
    #���ö������
    mk = imageio.v2.imread("bilibili.jpg")
    mword = len(top_words)
    font = "C:/Windows/Fonts/STXINGKA.TTF"
    w = WordCloud(font_path = font, mask = mk, width = 515, height = 455, min_font_size = 5, max_words = mword, background_color = 'white')
    #���ɴ���
    word_cloud = w.generate_from_frequencies(top_words)
    image_colors = ImageColorGenerator(mk)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.figure()
    plt.close()
    plt.imshow(word_cloud.recolor(color_func = image_colors))
    plt.axis("off")
    plt.show()
    #����ͼƬ
    #word_cloud.to_file("Bilibili_Bullet_Screen_High_Frequency_Words_Cloud.jpg")

#������
if __name__ == '__main__':

    #��������
    data = pd.read_csv("danmuku.csv", encoding = 'utf-8-sig')
    danmu_words = list(data['content'])[:10000]
    danmu_words.sort()
    stopwords = []
    f = open('stopwords_list.txt', 'r', encoding = 'utf-8-sig')
    for i in f.read().splitlines():
        stopwords.append(i)
    stopwords.sort()
    jieba.load_userdict("stopwords_list.txt")

    #�ִ�
    words_list = []
    words_count = {}
    for i in danmu_words:
        wordf = list(jieba.cut(i))
        worde = []
        for j in wordf:
            #����ͣ�ô�
            if j not in stopwords:
                worde.append(j)
                #ͳ�ƴ�Ƶ
                if j in words_count.keys():
                    words_count[j] += 1
                else:
                    words_count[j] = 1
        words_list.append(worde)

    #����ض���Ŀ�ĸ�Ƶ�ʺ͵�Ƶ��
    value, requested = 100, '0'
    print("num={} -> {}".format(value, word_search(value, requested)))

    #������ɸѡ
    words_feature = (word_screen(100, -1))
    print("term:{}".format(words_feature))

    #��Ļ��������ʾ
    words_vector = []
    words_vector_num = []
    for i in range(len(words_list)):
        word = words_list[i]
        if len(word) > 4:
            words_vector.append(word_vector(word))
            words_vector_num.append(i)
    #print(words_vector)

    #̽���������ƶ�
        #����һά�����������Ϊ��Ҫ�Ƚ��������ƶȵĵ�Ļ��������
    index_list1 = np.random.randint(0, len(words_vector), 5)
    index_list2 = np.random.randint(0, len(words_vector), 5)
    print(index_list1, index_list2)
        #�ֶ�ѡ����Ҫ�Ƚ��������ƶȵĵ�Ļ��������
        #index_list1 = np.array([1, 10, 100, 1000])
        #index_list2 = np.array([2, 20, 200, 2000])
    for i in range(len(index_list1)):
        idx_1, idx_2 = words_vector_num[index_list1[i]], words_vector_num[index_list2[i]]
        print("first matrix: the {} bullet chat is :{}  cut word is :{}  word vector is {}:".format(i+1, danmu_words[idx_1], words_list[idx_1], words_vector[index_list1[i]]))
        print("second matrix: the {} bullet chat is :{}  cut word is :{}  word vector is {}:".format(i+1, danmu_words[idx_2], words_list[idx_2],words_vector[index_list2[i]]))
    euclid_similarity = euclidean_similarity(index_list1, index_list2)
    cos_similarity = cosine_similarity(index_list1, index_list2)
    print("euclidean similarity matrix:{}\n cosine similarity matrix:{}".format(euclid_similarity, cos_similarity))

    #��Ƶ�ʿ��ӻ�
    words_top = sorted(words_count.items(), key = lambda x:x[1], reverse = True)
    top = 100
    draw_word_cloud(top, words_top)
