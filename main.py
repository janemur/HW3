# импорт библиотек
import re  # для использования регулярных выражений
import numpy as np  # для создания матрицы

corpus_ = [
    'Crock Pot Pasta Never boil pasta again',
    'Pasta Pomodoro Fresh ingredients Parmesan to taste'
]
corpus_words = []


class CountVectorizer:

    def __init__(self, corpus):
        # разделяем предложения на слова
        for i in corpus:
            splitted_words = re.split('[^A-Za-z]+', i.lower().strip())
            corpus_words.append(splitted_words)

        self.words = {}
        index = 0

        # Создаем словарь, в котором уникальные слова предложений – ключи, а их номер – значение
        for i in corpus_words:
            for sent in i:
                if sent not in self.words:
                    self.words[sent] = index
                    index += 1

        num_words = len(self.words)
        num_sentences = len(corpus)

        # создаем пустую матрицу размером кол-во предложений, на кол-во уникальных слов
        self.matrix = np.zeros((num_sentences, num_words))

        # заполняем матрицу нужными значениями
        for i in range(num_sentences):
            cur_word = corpus_words[i]
            for w in cur_word:
                self.matrix[i][self.words[w]] += 1

    def feature_names(self):
        return list(self.words.keys())

    def fit_transform(self):
        return self.matrix


vectorizer = CountVectorizer(corpus_)
print(vectorizer.feature_names())
print(vectorizer.fit_transform())

if __name__ == '__main__':
    CountVectorizer(corpus_)
