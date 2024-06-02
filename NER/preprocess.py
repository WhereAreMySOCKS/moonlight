"""dataset preprocess for NER task"""
import pickle

from NER.my_dataset import MyDataset
import re

MAX_LEN = 130

def read_data(file_path):
    """Read data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        text = ''
        labels = []
        for count,line in enumerate(lines):
            line = line.replace('\n','')
            if len(line) <= 1 :
                print(file_path, f" 第{count}行为空,row:{line}")
                continue
            char, label = line.split(' ')
            text += char
            labels.append(label)
            if len(text) != len(labels):
                print(f"第{count}行标签与文本不匹配,文本长度{len(text)},标签长度{len(labels)},文本:{text[0]},标签:{labels[0]}")

    return text,labels

def gen_label2id(labels):
    """Generate label2id dict and print label distribution"""
    label2id = {}
    for label in labels:
        if label not in label2id:
            label2id[label] = len(label2id)

    # Save label2id
    with open('./data/dataset/label2id.pkl', 'wb') as f:
        pickle.dump(label2id, f)
    return label2id

def label_to_id(labels, label2id):
    """Convert label to id"""
    label_ids = []
    for label in labels:
        label_ids.append(label2id[label])
    return label_ids

def gen_pair(mod,max_len):
    """
    通过句号分割文本，生成每个句子的文本和标签对。此步操作后句子通过padding固定为max_len
    :param mod:
    :return:
    """
    # 只能读取train中的标签
    train_data =  read_data('./data/data111/train.txt')
    l2id = gen_label2id(train_data[1])

    if mod == 'train':
        data = train_data
    elif mod ==  'test':
        data = read_data('./data/data111/test.txt')
    elif mod == 'val':
        data = read_data('./data/data111/dev_demo.txt')
    else:
        raise ValueError('mod error')
    sentence_list,data_segments, label_segments = [],[],[]

    # 确保标签的长度与文本中的字符数量相同
    text,label = data
    assert len(text) == len(label), f"标签数量{len(label)}与文本字符数量{len(text)}不匹配"
    # 使用正则表达式来分割句子，匹配中文句号
    sentences = re.split(r'。\s*', text)

    # 由于split会保留最后一个空字符串，需要将其移除
    if sentences and not sentences[-1]:
        sentences.pop()

    # 为每个句子分配标签
    tagged_sentences = []
    sentence_start_index = 0

    for i, sentence in enumerate(sentences):
        if sentence:  # 跳过空字符串
            # 计算当前句子的标签列表
            sentence_labels = label[sentence_start_index:sentence_start_index + len(sentence)]
            # 更新句子起始索引
            sentence_start_index += len(sentence) + 1  # 加1是因为split会将句号也计算在内
            tagged_sentences.append((sentence, sentence_labels))

    max_len_ = 0
    for i, (sentence, sentence_labels) in enumerate(tagged_sentences, 1):
        # 统计max-sentence长度
        if len(sentence) > max_len_:
            max_len_ = len(sentence)
    print(f"{mod} 中句子最大长度:", max_len_)

    for i, (sentence, sentence_labels) in enumerate(tagged_sentences, 1):
        assert len(sentence) == len(sentence_labels), f"第{i}个句子的标签数量{len(sentence_labels)}与文本字符数量{len(s_list)}不匹配"
        if len(sentence) <= max_len - 1 :
            # 用['pad']补齐句子长度到max_len
            s_list = ['CLS'] + list(sentence)
            sentence_labels = ['O'] + sentence_labels
            sentence_list.append( s_list + ['PAD'] * (max_len  - len(s_list)))
            sentence_labels = sentence_labels + ['O'] * (max_len - len(s_list))
        else:
            # 截断句子
            s_list = ['CLS'] + list(sentence)[:max_len - 1]
            sentence_labels = ['O'] + sentence_labels[:max_len - 1]
            sentence_list.append(s_list)
            sentence_labels = sentence_labels + ['O'] * (max_len - len(s_list))

        data_segments.append(sentence)
        label_segments.append(label_to_id(sentence_labels, l2id))

    return data_segments, sentence_list,label_segments

def create_vocab(data):
    """
    创建lstm和cnn使用的vocab
    :param data:
    :return: vocab {'char1':1,'char2':2,...}
    """
    data = set(list(''.join(data)))
    vocab = {'PAD': 0,'CLS':1,'SEP':2,'UNK':3}
    # 更新词典
    for word in data:
        vocab[word] = len(vocab)
    # 保存词典
    with open('./data/dataset/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    return vocab

def convert_char_to_id(word_list,vocab):
    """
    这里的字符是在不使用bert的情况下，将字符id输入embedding时使用。
    如果使用bert模型，就不需要这个id，使用bert的tokenizer获取id即可。
    """
    char_id = []
    for i in word_list:
        tmp = []
        for j in i:
            if vocab.get(j):
                tmp.append(vocab[j])
            else:
                tmp.append(vocab['UNK'])
        char_id.append(tmp)
    return char_id


if __name__ == '__main__':
    train_text,train_text_list,train_label = gen_pair('train',max_len = MAX_LEN)
    test_text,test_text_list,test_label = gen_pair('test',max_len = MAX_LEN)
    val_text,val_text_list,val_label = gen_pair('val',max_len = MAX_LEN)
    v = create_vocab(train_text)
    train_text_ids = convert_char_to_id(train_text_list,v)
    test_text_ids = convert_char_to_id(test_text_list,v)
    val_text_ids = convert_char_to_id(val_text_list,v)

    train_dataset = MyDataset(train_text,train_text_ids,train_label,MAX_LEN)
    test_dataset = MyDataset(test_text,test_text_ids,test_label,MAX_LEN)
    val_dataset = MyDataset(val_text,val_text_ids,val_label,MAX_LEN)

    #pkl保存dataset
    with open('./data/dataset/train.pkl','wb') as f:
        pickle.dump(train_dataset,f)
    with open('./data/dataset/test.pkl','wb') as f:
        pickle.dump(test_dataset,f)
    with open('./data/dataset/val.pkl','wb') as f:
        pickle.dump(val_dataset,f)

