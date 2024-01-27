"""
CS 481 Programming Assignment 2
Created by Jake Norbie on 3/27/2023
"""


def preprocess(ignore_step):
    # print message starting preprocessing
    ignore_msg = "STEMMING" if ignore_step else "NONE"
    print("Ignored pre-processing step: " + ignore_msg)

    # preprocessing the csv data
    punctuation_list = [".", ",", "!", "?"]
    df = pd.read_csv("Norbie_Jake_CS481_P02_data.csv")
    df["review"] = df["reviewText"].astype(str)
    df["review"] = df["review"].apply(
        lambda s: "".join([letter for letter in s if letter not in punctuation_list])
    )
    df["review"] = df["review"].apply(lambda s: s.strip().split())
    df = df[["review", "overall"]]

    # show graph of label distribution overall
    fig = plt.figure()
    plt.bar(df["overall"].value_counts().index, df["overall"].value_counts())
    plt.xlabel('"Overall" Label')
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Labels for Dataset")
    # plt.show()
    plt.close()

    # preprocessing the text
    stop_corpus = nltk.corpus.stopwords.words("english")

    #  lowercasing
    df["review"] = df["review"].apply(lambda s: [word.lower() for word in s])

    #  stop word removal
    df["review"] = df["review"].apply(
        lambda s: [word for word in s if word not in stop_corpus]
    )

    #  stemming
    if not ignore_step:
        ps = nltk.stem.PorterStemmer()
        df["review"] = df["review"].apply(lambda s: [ps.stem(word) for word in s])

    train(df)


def feature_generation(data, bag_of_words_map):
    feature_list = [0] * len(bag_of_words_map)
    for word in data:
        try:
            feature_list[bag_of_words_map.index(word)] += 1
        except KeyError or ValueError:
            continue
    return feature_list


def train(df):
    # print training start message
    print("Training classifier...")

    # test-train split
    train_data = df.sample(frac=0.8)
    test_data = df.drop(train_data.index)
    labels = train_data["overall"].unique()

    # convert label to onehot
    lb = LabelBinarizer().fit(train_data["overall"])
    test_data["onehot_overall"] = list(lb.transform(test_data["overall"]))

    # show graph of label distributions of training and testing datasets
    fig = plt.figure()
    plt.bar(
        train_data["overall"].value_counts().index - 0.2,
        train_data["overall"].value_counts(),
        0.4,
        label="Train Dataset",
    )
    plt.bar(
        test_data["overall"].value_counts().index + 0.2,
        test_data["overall"].value_counts(),
        0.4,
        label="Test Dataset",
    )
    plt.xlabel('"Overall" Label')
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Labels for Train/Test Dataset")
    plt.legend()
    # plt.show()
    plt.close()

    # assemble bag of words
    bag_of_words_map = []
    for _, row in train_data.iterrows():
        for word in row["review"]:
            if word not in bag_of_words_map:
                bag_of_words_map.append(word)

    # generate features for each review/document
    train_data["feature"] = train_data["review"].apply(
        lambda x: feature_generation(x, bag_of_words_map)
    )

    # create parameter dicts
    #  create label probability dictionary
    label_prob_dict = {}
    for label in labels:
        label_prob_dict[label] = (
            train_data[train_data["overall"] == label].shape[0] / train_data.shape[0]
        )

    #  create label word counts and "word | label" probabilities
    label_word_counts = {}
    word_probabilities = {}
    for label in labels:
        labeled_train_data = train_data[train_data["overall"] == label]
        word_counts = labeled_train_data["review"].apply(len)
        label_word_counts[label] = word_counts.sum()
        # find total word usage within labeled_train_data + 1,
        # and divide it by label_word_counts + len(labels) before adding it
        word_probabilities[label] = []
        for idx, word in enumerate(bag_of_words_map):
            word_probabilities[label].append(
                (sum(labeled_train_data["feature"].apply(lambda x: x[idx])) + 1)
                / (label_word_counts[label] + len(bag_of_words_map))
            )

    test(test_data, bag_of_words_map, labels, label_prob_dict, word_probabilities)


def predict(data, labels_map, label_probs, word_probs):
    # a list mapped by labels map to show probabilities
    predict_scores = []
    # test the probability of each label, and find the maximum probability
    for label in labels_map:
        # get general label probability
        label_prob = label_probs[label]

        # find bayes probabilities of vectorized words
        vector_probs = 1
        for probability, occurences in zip(word_probs[label], data):
            if occurences != 0:
                vector_probs *= probability * occurences

        # get final naive bayes probability
        predict_scores.append(vector_probs * label_prob)

    # return result
    return predict_scores


def test(test_data, bag_of_words_map, labels_map, label_probs, word_probs):
    # print message
    print("Testing classifier...")

    # generate features for the testing document
    test_data["feature"] = test_data["review"].apply(
        lambda x: feature_generation(x, bag_of_words_map)
    )

    # apply prediction function to features
    test_data["predict_scores"] = test_data["feature"].apply(
        lambda x: predict(x, labels_map, label_probs, word_probs)
    )
    test_data["predict"] = test_data["predict_scores"].apply(
        lambda l: labels_map[l.index((max(l)))]
    )

    # graph the results in a confusion matrix
    # craft confusion matrix visual
    cm = confusion_matrix(test_data["overall"], test_data["predict"], labels=labels_map)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels_map)
    disp.plot()
    plt.title("Confusion Matrix w/o Stemming")
    # plt.show()
    plt.close()

    # incredibly stupidly complicated way to creating a roc curve
    onehot_labels = np.array(test_data["onehot_overall"].values.tolist())
    label_scores = np.array(test_data["predict_scores"].values.tolist())
    fig, ax = plt.subplots(figsize=(6, 6))
    for color, label_pos in zip(
        ["red", "orange", "yellow", "green", "blue"], range(len(labels_map))
    ):
        RocCurveDisplay.from_predictions(
            onehot_labels[:, label_pos],
            label_scores[:, np.where(labels_map == label_pos + 1)[0]],
            name=f"ROC curve for {label_pos + 1}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves w/o Stemming")
    # plt.show()

    # calculate and print various metrics
    for idx, label in enumerate(labels_map):
        tp = cm[idx][idx]
        fn = sum(np.append(cm[idx][0:idx], cm[idx][idx + 1 :]))
        fp = sum([row[idx] if i != idx else 0 for i, row in enumerate(cm)])
        tn = sum(
            [
                sum([element if j != idx else 0 for j, element in enumerate(row)])
                if i != idx
                else 0
                for i, row in enumerate(cm)
            ]
        )
        print("\nTest results / metrics for label '" + str(label) + "':")
        print("-" * 20)
        print("Number of true positives: " + str(tp))
        print("Number of false negatives: " + str(fn))
        print("Number of false positives: " + str(fp))
        print("Number of true negatives: " + str(tn))
        print("Sensitivity (recall): " + str(tp / (tp + fn)))
        print("Specificity: " + str(tn / (tn + fp)))
        try:
            print("Precision: " + str(tp / (tp + fp)))
        except RuntimeWarning():
            print("Precision: nan")
        print("Negative predictive value: " + str(tn / (tn + fn)))
        print("Accuracy: " + str((tp + tn) / (tp + fn + fp + tn)))
        print("F-Score: " + str(tp / (tp + (0.5 * (fp + fn)))))

    # begin predicting new sentences
    input_predict_loop(bag_of_words_map, labels_map, label_probs, word_probs)


def input_predict_loop(bag_of_words_map, labels_map, label_probs, word_probs):
    print("")
    sentence = input("Enter your sentence: ")

    # preprocess sentence
    stop_corpus = nltk.corpus.stopwords.words("english")
    ps = nltk.stem.PorterStemmer()
    punctuation_list = [".", ",", "!", "?"]
    sentence = "".join(
        [letter for letter in sentence if letter not in punctuation_list]
    )
    sentence = [word.lower() for word in sentence]
    sentence = [word for word in sentence if word not in stop_corpus]
    sentence = [ps.stem(word) for word in sentence]

    # generate features for the training document
    feature_vector = feature_generation(sentence, bag_of_words_map)

    # apply prediction function to features
    predict_scores = predict(feature_vector, labels_map, label_probs, word_probs)
    guess = labels_map[predict_scores.index((max(predict_scores)))]

    # printing format
    print("\nSentence S:\n")
    print(sentence)
    print("\nwas classified as " + str(guess) + ".")

    for idx, label in enumerate(labels_map):
        print("P(" + str(label) + " | S) = " + str(predict_scores[idx]))

    # continue message loop
    while 1:
        continue_msg = input("Do you want to enter another sentence [Y/N]? ")
        if continue_msg == "Y":
            break
        elif continue_msg == "N":
            return
        else:
            print("Enter a valid prompt")

    input_predict_loop(bag_of_words_map, labels_map, label_probs, word_probs)


def cla():
    args = sys.argv
    ignore = False
    if len(args) == 2 and args[1] == "YES":
        ignore = True

    # print beginning of program
    print("Norbie, Jake, A20459012 Solution:")

    preprocess(ignore)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import nltk
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        RocCurveDisplay,
    )
    from sklearn.preprocessing import LabelBinarizer
    import sys

    nltk.download("stopwords")

    cla()
