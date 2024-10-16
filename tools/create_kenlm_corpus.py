import json
import random
from tqdm import tqdm


def convert_line(line: list) -> list:
    sep_idxs = [i for i, c in enumerate(line) if c == '¦']
    sep_idxs = [-1] + sep_idxs + [len(line)]
    line = [line[(i + 1):j] for i, j in zip(sep_idxs[:-1], sep_idxs[1:])]

    random.shuffle(line)
    for i in range(len(line) - 1):
        if len(line[i]) > 30 or random.randint(1, 5) == 1:
            # randomly keep '¦' of 1/5 of the words
            line[i].append('¦')
    line = [c for word in line for c in word]
    return line


def create_data():
    with open('../dataset/manifest.train', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fp = open('../dataset/corpus.txt', 'w', encoding='utf-8')
    for line in tqdm(lines):
        data = json.loads(line)
        text = data['text']
        text = convert_line(text)
        text = ' '.join(text)
        fp.write(f'{text}\n')
    fp.close()


if __name__ == '__main__':
    create_data()
