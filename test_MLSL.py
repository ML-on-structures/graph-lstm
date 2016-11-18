import bipartite_user_item_reviews as bp
import mlsl as ml
import random
import numpy as np
import unittest

def test_model(mlsl_model, test_set):
    guesses = 0
    hits = 0
    found = {}
    missed = {}
    misclassified = {}
    for item in test_set:
        Y = softmax(mlsl_model.forward_propagation(item))
        if Y is None:
            continue
        print Y
        predicted_label = Y.argmax()
        real_label = item.get_label()
        print "Predicted label ", predicted_label , " real label", real_label
        guesses += 1
        hits += 1 if predicted_label == real_label else 0
        if predicted_label == real_label:
            if real_label not in found:
                found[real_label] = 1
            else:
                found[real_label] += 1
        if predicted_label != real_label:
            if real_label not in missed:
                missed[real_label] = 1
            else:
                missed[real_label] += 1
            if predicted_label not in misclassified:
                misclassified[predicted_label] = 1
            else:
                misclassified[predicted_label] += 1
    print "LSTM results"
    print "============================================================="
    print "Predicted correctly ", hits , "over ", guesses, " instances."
    recall_list = []
    recall_dict = {}
    precision_dict = {}
    found_labels = set(found.keys())
    missed_labels = set(missed.keys())
    all_labels = found_labels.union(missed_labels)
    for label in all_labels:
        no_of_finds = float((0 if label not in found else found[label]))
        no_of_missed = float((0 if label not in missed else missed[label]))
        no_of_misclassified = float((0 if label not in misclassified else misclassified[label]))
        recall =  no_of_finds / (no_of_finds + no_of_missed)
        precision = no_of_finds / (no_of_finds + no_of_misclassified)
        recall_dict[label] = recall
        precision_dict[label] = precision
        recall_list.append(recall)
    avg_recall = np.mean(recall_list)
    print "Average recall ", np.mean(recall_list)
    if len(all_labels) == 2: # compute F-1 score for binary classification
        for label in all_labels:
            print "F-1 score for label ", label, " is : ",
            print 2 * (precision_dict[label] * recall_dict[label]) / (precision_dict[label] + recall_dict[label])
    return avg_recall

def create_synthetic_graph_with_informative_extra_feature(no_of_items, no_of_users, no_of_votes, min_grade, max_grade, min_delay, max_delay, threshold, exclude_true_grade_from_random_answers = False, seed = 500):
    """
    Creates random graph.
    Adds extra feature in review: if value is above threshold, the user performing the review is always truthful.
    Intended for testing the MLSL network across multiple levels by feeding extra information selectively to particular levels.
    When exclude_true_grade_from_random_answers is True the true grade is excluded from the possible random grades, i.e. the random review grade is alway not the true grade.

    """
    g = bp.Graph()
    random.seed(seed)
    # add items and their inherent random grades to the graph
    # inherent grades are uniformly distributed between min_grade and max_grade
    item_ids = [str(i) for i in range(no_of_items)]
    for i in item_ids:
        g.add_item(i, random.randint(min_grade,max_grade))
    # item_ids is a list that contains the ids of items that are still alive, i.e.
    # the ones that have received less than no_of_votes votes.
    # Initially before users cast their votes, the list contains all items.
    user_ids = [str(u) for u in range(no_of_users)]
    for u in user_ids: # iteration over users and random picking of items to vote
        user_votes = 0
        items_complete_with_votes = []
        # shuffling items so that user picks items to vote randomly
        random.shuffle(item_ids)
        extra_informative = random.randint(min_delay, max_delay)
        for i in item_ids:
            # if item has over the max number of votes or user has already voted on it
            # then continue to next item
            if g.get_no_of_votes(i) >= no_of_votes:
                items_complete_with_votes.append(i)
                continue
            if g.has_voted(u,i):
                continue
            # all clear to create review for item
            possible_random_grades = range(min_grade, max_grade + 1)
            if exclude_true_grade_from_random_answers:
                possible_random_grades.remove(g.get_item(i).inherent)
            review = bp.Review( grade = g.get_item(i).inherent if extra_informative>threshold else random.choice(possible_random_grades), extra_informative_feature = extra_informative)
            g.add_review(u,i,review)
            user_votes += 1
            # if item has exceeded max number of votes add it to the list for removal
            if g.get_no_of_votes(i) >= no_of_votes:
                items_complete_with_votes.append(i)
            # if user has cast more than max number of votes, break loop and continue to next user
            if user_votes == no_of_votes:
                break
        # remove items with more than max number of votes from item_ids list to avoid iterating over them in the future (makes execution faster)
        for i in items_complete_with_votes:
            item_ids.remove(i)
    return g


def test_for_multiple_layers(print_graph = False, max_depth = 0, informative_features = None):
    votes = 3
    g = create_synthetic_graph_with_informative_extra_feature(no_of_items = 3000, no_of_users = 3000,
                                                               no_of_votes = votes, min_grade = 0, max_grade = 10,
                                                               min_delay= 0.0, max_delay = 1000.0,
                                                               threshold = 700.0)
    random.seed(940)
    itemList = list(g.items)
    if print_graph:
        for u in g.users:
            print "User ", u.name, "voted items:"
            for i in u.reviews:
                print "Item ", i.id, "Inherent grade:", i.inherent, "User grade:", u.reviews[i].grade, "Extra feature: ", u.reviews[i].extra_informative_feature
    instance_list = []
    counter = 0
    for i in itemList:
        new_root = ml.InstanceNode(label = i.inherent)
        build_unfolding(0, max_depth, i, new_root, informative_features)
        new_root.set_label(i.inherent)
        instance_list.append(new_root)
        counter +=1
        if counter % 200 ==0:
            print "Created unfolding for ", counter, "items."
    OUTPUT_SIZES = [11, 2, 2]
    INPUT_SIZES = [11 + (1 if informative_features[0] == "include" else 0),11 + (1 if informative_features[1] == "include" else 0),
                   11 + (1 if informative_features[2] == "include" else 0)]
    LEARNING_RATE_VECTOR = [0.05,0.1, 4.5]
    LEARNING_METHOD_VECTOR = ["steady_rate", "steady_rate","steady_rate"]
    #LEARNING_METHOD_VECTOR = ["momentum", "momentum", "momentum"]
    #LEARNING_METHOD_VECTOR = ["adadelta", "adadelta", "adadelta"]
    MOMENTUM_VECTOR = [0.01, 0.01, 0.01]
    ADADELTA_VECTOR = [{"learning_factor" : 1.0, "epsilon" : 0.001, "decay" : 0.95}, {"learning_factor" : 1.0, "epsilon" : 0.001, "decay" : 0.95}, {"learning_factor" : 1.0, "epsilon" : 0.001, "decay" : 0.95}]
    OBJECTIVE_FUNCTION = "softmax_classification"
    mlsl_model = ml.MLSL(max_depth + 1, output_sizes= OUTPUT_SIZES[:max_depth + 1], node_feature_sizes= INPUT_SIZES[:max_depth + 1], learning_rate_vector= LEARNING_RATE_VECTOR[:max_depth + 1], learning_method_vector= LEARNING_METHOD_VECTOR[:max_depth + 1])
    random.shuffle(instance_list)
    training_set = instance_list[0:2000]
    test_set = instance_list[2000:3000]
    print "Training starts for ", max_depth + 1, " levels"
    train_model_force_balance(mlsl_model, training_set, num_instances = 50000,
                                          max_depth= max_depth, objective_function= OBJECTIVE_FUNCTION,
                                          learning_rate_vector= LEARNING_RATE_VECTOR, learning_method_vector = LEARNING_METHOD_VECTOR,
                                          momentum_vector= MOMENTUM_VECTOR, adadelta_parameters = ADADELTA_VECTOR)
    return test_model(mlsl_model, test_set)

def build_unfolding(current_depth, max_depth, bipartite_node, tree_node, informative_features = None, parent_user_informative_feature = None):
    for c in bipartite_node.reviews:
        number_of_features = 12 if informative_features[current_depth] == "include" else 11
        feature_vector = np.zeros(number_of_features)
        feature_vector[bipartite_node.reviews[c].grade] = 1.0
        if current_depth == 0: # nake explicit honesty feature to feed to the 3rd level for 3 level training test
            if bipartite_node.inherent == bipartite_node.reviews[c].grade:
                honesty = 1.0
            else:
                honesty = 0.0
        extra_feature = bipartite_node.reviews[c].extra_informative_feature / 1000.0
        if number_of_features == 12:
            feature_vector[11] = extra_feature if current_depth < 2 else parent_user_informative_feature
        child_node = ml.InstanceNode(feature_vector = feature_vector.copy())
        if current_depth < max_depth:
            build_unfolding(current_depth + 1, max_depth, bipartite_node = c, tree_node= child_node, informative_features = informative_features,
                            parent_user_informative_feature = honesty if current_depth == 0 else parent_user_informative_feature)
        tree_node.children.append(child_node)


"""
trains MLSL with stochastic gradient descent
by imposing class balance, i.e. shows equal number of examples of all classes to the network during training
"""
def train_model_force_balance(mlsl_model, training_set, num_instances, max_depth, objective_function, learning_rate_vector, learning_method_vector, momentum_vector = None, adadelta_parameters = None):
    counter = 0
    if num_instances == 0:
        return
    for item in get_balanced_training_set(training_set, mlsl_model.output_sizes[0]):
        if item.get_number_of_children() == 0:
            continue
        target = np.zeros((1,mlsl_model.output_sizes[0]))
        target[0,item.get_label()] = 1.0
        mlsl_model._reset_learning_parameters()
        Y = softmax(mlsl_model.forward_propagation(item))
        mlsl_model.backward_propagation(item, Y - target)
        counter += 1
        if counter % 1000 == 0:
            print "Training has gone over", counter, " instances.."
        if counter == num_instances:
            break

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

""" Generator that returns items from training set
    equally balanced among classes"""
def get_balanced_training_set(training_set, no_of_classes):
    # make bucket of classes to sample from
    buckets = {}
    buckets_current_indexes ={}
    for i in range(0, no_of_classes):
        buckets[i] = []
        buckets_current_indexes[i] = 0
    for item in training_set:
        category = item.get_label()
        buckets[category].append(item)
    while True:
        for i in range(0,no_of_classes):
            if len(buckets[i]) == 0: # if a class has no representatives, continue
                continue
            if buckets_current_indexes[i] == len(buckets[i]):
                buckets_current_indexes[i] = 0
            yield buckets[i][buckets_current_indexes[i]]
            buckets_current_indexes[i] += 1
            

class SimpleLearningTest(unittest.TestCase):

    def test_graph(self):
        # test for 1 level
        # by changing 'exclude' to 'includde' we include the informative feature
        # and expect performance to improve
        first_level_performance = test_for_multiple_layers(print_graph= False, max_depth = 0, informative_features = ["exclude", "NA", "NA"])
        first_level_additional_feature_performance = test_for_multiple_layers(print_graph= False, max_depth = 0, informative_features = ["include","NA","NA"])

        # test for 2 levels
        # the 2 level (max_depth = 1) beats the 1 level as it can learn the informative feature at the second level
        second_level_no_additional_performance = test_for_multiple_layers(print_graph= False, max_depth = 1, informative_features = ["exclude", "exclude", "NA"])
        second_level_additional_performance = test_for_multiple_layers(print_graph= False, max_depth = 1, informative_features = ["exclude", "include", "NA"])

        # test for 3 levels
        third_level_additional_performance = test_for_multiple_layers(print_graph= False, max_depth = 2, informative_features = ["exclude", "exclude", "include"])

        print "\n\n\nAggregate performance comparison, test results"
        print "----------------------------------------------"
        print "1-MLSL performance, no additional informative feature : ", first_level_performance
        print "1-MLSL performance, with additional informative feature :", first_level_additional_feature_performance
        print "-----"
        print "Additional feature enhances performance -- training OK!" if first_level_additional_feature_performance> first_level_performance  else "Not OK"
        print "-----"
        print "2-MLSL performance, no additional informative feature :", second_level_no_additional_performance
        print "2-MLSL performance, additional informative feature fed to second level *only* :", second_level_additional_performance
        print "-----"
        print "Additional feature at second level enhances performance -- second level training OK!" if second_level_additional_performance> second_level_no_additional_performance  else "Not OK"
        print "-----"
        print "3-MLSL performance, additional informative feature on parent user honesty fed to third level *only* :", third_level_additional_performance
        print "-----"
        print "Additional feature at third level enhances performance -- third level training OK!" if third_level_additional_performance> second_level_no_additional_performance  else "Not OK"
        print "-----"
        self.assertGreater(first_level_additional_feature_performance, first_level_performance,"1 Level training not OK, retrain!")
        self.assertGreater(second_level_additional_performance, second_level_no_additional_performance,"2 level training not OK, retrain!")
        self.assertGreater(third_level_additional_performance,second_level_no_additional_performance, "3 level training not OK, retrain!")
        # If one occasionally gets Not OK results, retrain as the random initiliazation of the weights can sometimes trap the network

if __name__ == '__main__':
    unittest.main()