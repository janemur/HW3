# импорт библиотек
import re  # для использования регулярных выражений


class CountVectorizer:
    """ Класс CountVectorizer берет на вход лист с текстовыми значениями (предложениями) и выводит уникальные слова в
    этих предложениях и матрицу из каких слов состоит предложение и как часто они там встречаеются
    """

    def __init__(self):
        self.matrix = []  # матрица значений
        self.corpus_words = []  # складываем сюда разделенные слова
        self.words = {}  # будут хранится уникальные слова

    def fit_transform(self, corpus):
        """ Выводит матрицу со значениями, как часто каждое слово из словаря встречается в данном предложении,
        если слово отсутсвует ставится 0 """

        # разделяем предложения на слова
        for i in corpus:
            splitted_words = re.split('[^A-Za-z]+', i.lower().strip())
            self.corpus_words.append(splitted_words)

        index = 0

        # Создаем словарь, в котором уникальные слова предложений – ключи, а их номер – значение
        for i in self.corpus_words:
            for sent in i:
                if sent not in self.words:
                    self.words[sent] = index
                    index += 1

        num_words = len(self.words)
        num_sentences = len(corpus)

        # Заполняем матрицу нулевыми значениями

        for i in range(num_sentences):
            self.matrix.append([0] * num_words)

        # заполняем матрицу нужными значениями
        for i in range(num_sentences):
            cur_word = self.corpus_words[i]
            for w in cur_word:
                self.matrix[i][self.words[w]] += 1

        return self.matrix

    def feature_names(self):
        """ Выводит ключи словаря, они же -- уникальные слова, которые встречаются во всех данных предложениях"""
        return list(self.words.keys())


if __name__ == '__main__':
    corpus_ = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(corpus_))
    print(vectorizer.feature_names())
