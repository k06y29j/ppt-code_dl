"""文本预处理"""
import collections
import re
from d2l import torch as d2l    

"""读取数据集"""
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine(): 
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 删除空格和换行符，并将所有字符转换为小写,^A-Za-z表示只保留字母字符
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

"""词元化"""
def tokenize(lines, token='word'):
    """将文本行列表转换为词元列表"""
    if token == 'word':
        # 使用空格分隔单词
        return [line.split() for line in lines]
    elif token == 'char':
        # 将每个字符作为一个词元
        return [list(line) for line in lines]
    else:
        print('错误的token类型:', token)
tokens=tokenize(lines)
for i in  range(11):
    print(tokens[i])

print("------------词表-----------------")
class Vocab: 
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        #reverse=True表示按频率降序排列
        # items()返回一个包含所有键值对的列表
        # sorted()函数对列表进行排序
        # key=lambda x: x[1]表示按频率（第二个元素）排序
        # 结果是一个列表，包含所有词元及其频率
        # 例如：[('the', 100), ('a', 80), ('and', 60), ...]
        # 这里的counter是一个字典，键是词元，值是频率
        # sorted()函数将字典转换为一个按频率排序的列表
        # 例如：[('the', 100), ('a', 80), ('and', 60), ...]

        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 将保留的词元添加到词表中
        # idx_to_token是一个列表，包含所有词元
        # token_to_idx是一个字典，键是词元，值是索引
        # 例如：{'the': 0, 'a': 1, 'and': 2, ...}
        # unk表示未知词元的索引
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    def unk(self):  # 未知词元的索引为0
        return 0

    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab= Vocab(tokens)
print(f'词表大小: {len(vocab)}')
print(f'前10个词元: {vocab.idx_to_token[:10]}')

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))