import tensorflow as tf
import os


def build_vocab(corpus_file, vocab_file, tag_file):
    """构建词典。"""
    words, tags = set(), set()

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            try:
                line = line.strip()
                word, tag = line.split()
                words.add(word)
                tags.add(tag)
            except Exception:
                print(line.split())

    if not os.path.exists(vocab_file):
        with open(vocab_file, "w") as f:
            for word in ["<UKN>"] + list(words):
                f.write(word+"\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "M": 2,
        "E": 3,
    }

    tags = sorted(
        list(tags),
        key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
    )

    if not os.path.exists(tag_file):
        with open(tag_file, "w") as f:
            for tag in ["<UKN>"] + tags:
                f.write(tag+"\n")


def read_vocab(vocab_file):
    """读取词典。"""
    with open(vocab_file, "r", encoding="utf-8") as f:
        gen = (line.strip() for line in f.readlines())

    vocab2id, id2vocab = dict(), dict()
    for index, line in enumerate(gen):
        vocab2id[line] = index
        id2vocab[index] = line

    return vocab2id, id2vocab


def tokenize(filename, vocab2id, tag2id):
    """数据预处理。"""
    contents, labels = list(), list()
    content, label = list(), list()

    with open(filename, "r", encoding="utf-8") as f:
        lines = [elem.strip() for elem in f.readlines()]

    for line in lines:
        try:
            if line != "":
                word, tag = line.split()
                content.append(vocab2id.get(word, 0))
                label.append(tag2id.get(tag, 0))
            else:
                if content and label:
                    contents.append(content)
                    labels.append(label)
                content = list()
                label = list()
        except Exception:
            content = list()
            label = list()

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding="post")
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding="post")
    labels = tf.one_hot(labels, len(tag2id))

    return contents, labels


tag_check = {
    "M": ["B", "M"],
    "E": ["B", "M"],
}


def check_label(front_label, follow_label):
    """标签检查。"""
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("M-") or follow_label.startswith("E-")) and \
            front_label.endswith(follow_label.split("-")[1]) and \
            front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True

    return False


def format_result(chars, tags):
    """结果格式化。"""
    entities, entity = list(), list()
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(
            tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = list()
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = list()
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result
