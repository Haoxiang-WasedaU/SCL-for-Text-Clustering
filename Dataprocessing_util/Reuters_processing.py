from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
# from keras.utils.np_utils import to_categorical
(train_data,train_labels), (test_data, test_labels) = reuters.load_data(test_split=0.2)
class0,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,\
class11,class12,class13,class14,class15,class16,class17,class18,class19,class20,\
class21,class22,class23,class24,class25,class26 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
class27,class28,class29,class30,\
class31,class32,class33,class34,class35,class36,class37,class38,class39 = 0,0,0,0,0,0,0,0,0,0,0,0,0
class40,class41,class42,class43,class44,class45,class46 = 0, 0, 0, 0, 0, 0, 0
counters = [0 for i in range(45)]
for i in range(len(train_labels)):
    if train_labels[i] == 0:
        class0 += 1
    if train_labels[i]==1:
        class1 += 1
    if train_labels[i]==2:
        class2+=1
    if train_labels[i]==3:
        class3+=1
    if train_labels[i]==4:
        class4+=1
    if train_labels[i]==5:
        class5+=1
    if train_labels[i]==6:
        class6+=1
    if train_labels[i]==7:
        class7+=1
    if train_labels[i]==8:
        class8+=1
    if train_labels[i]==9:
        class9+=1
    if train_labels[i]==10:
        class10+=1
    if train_labels[i]==11:
        class11+=1
    if train_labels[i]==12:
        class12+=1
    if train_labels[i]==13:
        class13+=1
    if train_labels[i]==14:
        class14+=1
    if train_labels[i]==15:
        class15+=1
    if train_labels[i]==16:
        class16+=1
    if train_labels[i]==17:
        class17+=1
    if train_labels[i]==18:
        class18+=1
    if train_labels[i]==19:
        class19+=1
    if train_labels[i]==20:
        class20+=1
    if train_labels[i]==21:
        class21+=1
    if train_labels[i]==22:
        class22+=1
    if train_labels[i]==23:
        class23+=1
    if train_labels[i]==24:
        class24+=1
    if train_labels[i]==25:
        class25+=1
    if train_labels[i]==26:
        class26+=1
    if train_labels[i]==27:
        class27+=1
    if train_labels[i]==28:
        class28+=1
    if train_labels[i]==29:
        class29+=1
    if train_labels[i]==30:
        class30+=1
    if train_labels[i]==31:
        class31+=1
    if train_labels[i]==32:
        class32+=1
    if train_labels[i]==33:
        class33+=1
    if train_labels[i]==34:
        class34+=1
    if train_labels[i]==35:
        class35+=1
    if train_labels[i]==36:
        class36+=1
    if train_labels[i]==37:
        class37+=1
    if train_labels[i]==38:
        class38+=1
    if train_labels[i]==39:
        class39+=1
    if train_labels[i]==40:
        class40+=1
    if train_labels[i]==41:
        class41+=1
    if train_labels[i]==42:
        class42+=1
    if train_labels[i]==43:
        class43+=1
    if train_labels[i]==44:
        class44 += 1
    if train_labels[i] == 45:
        class45 += 1
    if train_labels[i] == 46:
        class46 += 1
print(class0,class1,class2,class3,class4,class5,class6,class7,class8,class9,class10)
print(class11,class12,class13,class14,class15,class16,class17,class18,class19,class20)
print(class21,class22,class23,class24,class25,class26,class27,class28,class29,class30)
print(class31,class32,class33,class34,class35,class36,class37,class38,class39,class40)
print(class41,class42,class43,class44,class45,class46)
print(class1+class3+class4+class8+class10+class11+class13+class16+class19+class20)
final_data_label=[1,3,4,8,10,11,13,16,19,20]
print(final_data_label)
word_index= reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire2222 = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decoded_newswire = ' '.join([reverse_word_index.get(i, '?') for i in train_data[0]])
train_origin_data=[]
one_hot_train_labels = to_categorical(train_labels)
for data_piece in range(len(train_data)):
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '') for i in train_data[data_piece]])
    train_origin_data.append(decoded_newswire)
with open('Reuter_train_data', 'wb') as f:
    for q in range(len(train_origin_data)):
        f.write(train_origin_data[q].strip().encode()+'\t'.encode()+str(train_labels[q]).encode())
        f.write('\n'.encode())
f.close()
origin_text,text_label = [],[]
for line in open('Reuter_train_data', 'r').read().strip().split('\n'):
    text1, label = line.split('\t', 1)
    if int(label) in final_data_label:
        origin_text.append(text1)
        text_label.append(label)


with open('Reuter_final_data', 'wb') as f:
    for q in range(len(origin_text)):
        f.write(origin_text[q].strip().encode()+'\t'.encode()+text_label[q].encode())
        f.write('\n'.encode())