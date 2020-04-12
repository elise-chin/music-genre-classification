# Scoring
# Quantification de la qualite des predictions

def accuracy_score(guessed_labels, true_labels):
        """Computes the accuracy of the random forest.
        
        Arguments:
            guessed_labels {list} -- the list of labels guessed by the decision tree
            true_labels {list} -- the list of the real labels
        
        Returns:
            float -- the accuracy
        """
        score = 0
        n = len(guessed_labels)
        for i in range(n):
            score += (int(guessed_labels[i]) == int(true_labels[i]))
        score = score/n
        return score
