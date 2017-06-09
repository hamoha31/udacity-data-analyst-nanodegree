import sys
import pickle

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from time import time

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

#FUNCTIONS
def data_analysis():
    #Function for exploring the data.
    employee_names = data_dict.keys()
    employee_features = data_dict[employee_names[0]]
    number_poi, miss_email_poi = poi_missing_email_info()

    print 'Number of employees: ', len(employee_names)
    print 'Number of POI: ', number_poi

    #Removing outliers and updating NaN values.
    features_with_nan = fill_nan_values()

    # Remove outlier 'THE TRAVEL AGENCY IN THE PARK' since that is not a person.
    # Remove outlier 'TOTAL', determined from graph. and odd person value
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    # show_scatter_plot('salary', 'bonus')

def poi_missing_email_info():
    # Find total count and values of POI with missing or no to/from email information
    poi_count = 0
    poi_keys = []
    for person in data_dict.keys():
        if data_dict[person]["poi"]:
            poi_count += 1
            poi_keys.append(person)

    poi_missing_emails = []
    for poi in poi_keys:
        if (data_dict[poi]['to_messages'] == 'NaN' and data_dict[poi]['from_messages'] == 'NaN') or \
            (data_dict[poi]['to_messages'] == 0 and data_dict[poi]['from_messages'] == 0):
            poi_missing_emails.append(poi)

    return poi_count, poi_missing_emails 

def fill_nan_values():
    # Update NaN values with 0 except for email address.
    nan_values = {}
    employee_names = data_dict.keys()
    employee_features = data_dict[employee_names[0]]

    for feature in employee_features:
        nan_values[feature] = 0
    for name in employee_names:
        for feature in employee_features:
            if feature != 'email_address' and data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0
                nan_values[feature] += 1

    return nan_values

def show_scatter_plot(x, y):
    # Create scatter plot and show
    features = ['poi', x, y]
    data = featureFormat(data_dict, features)

    for point in data:
        x = point[1]
        y = point[2]
        plt.scatter(x, y, color='b', marker=".")
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def email_fractions():
    # Create new features for possible use in feature selection
    employee_names = data_dict.keys()

    for person in employee_names:
        to_poi = float(data_dict[person]['from_this_person_to_poi'])
        from_poi = float(data_dict[person]['from_poi_to_this_person'])
        to_msg_total = float(data_dict[person]['to_messages'])
        from_msg_total = float(data_dict[person]['from_messages'])

        if from_msg_total > 0:
            data_dict[person]['to_poi_fraction'] = to_poi / from_msg_total
        else:
            data_dict[person]['to_poi_fraction'] = 0

        if to_msg_total > 0:
            data_dict[person]['from_poi_fraction'] = from_poi / to_msg_total
        else:
            data_dict[person]['from_poi_fraction'] = 0

        # fraction of your salary represented by your bonus (or something like that)
        person_salary = float(data_dict[person]['salary'])
        person_bonus = float(data_dict[person]['bonus'])
        if person_salary > 0 and person_bonus > 0:
            data_dict[person]['salary_bonus_fraction'] = data_dict[person]['salary'] / data_dict[person]['bonus']
        else:
            data_dict[person]['salary_bonus_fraction'] = 0

    # Add new feature to features_list
    features_list.extend(['to_poi_fraction', 'from_poi_fraction', 'salary_bonus_fraction'])


def classifiers(classifier_type, k_best_features, features):
    data = featureFormat(my_dataset, features, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    sss = StratifiedShuffleSplit(labels, 500, test_size=0.5, random_state=40)

    # Build pipeline.
    k_best_features = SelectKBest(k=k_best_features)
    scaler = MinMaxScaler()
    classifier = set_classifier(classifier_type)
    pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', k_best_features), (classifier_type, classifier)])

    # Set parameters.
    parameters = []
    if classifier_type == 'random_forest':
        parameters = dict(random_forest__n_estimators=[25, 50],
                          random_forest__min_samples_split=[2, 3, 4],
                          random_forest__criterion=['gini', 'entropy'])
    if classifier_type == 'logistic_reg':
        parameters = dict(logistic_reg__class_weight=['balanced'],
                          logistic_reg__solver=['liblinear'],
                          logistic_reg__C=range(1, 5))
    if classifier_type == 'decision_tree':
        parameters = dict(decision_tree__min_samples_leaf=range(1, 5),
                          decision_tree__mx_depth=range(1, 5),
                          decision_tree__class_weight=['balanced'],
                          decision_tree__criterion=['gini', 'entropy'])
    if classifier_type == 'gaussian_nb':
    	parameters = dict(gaussian_nb__theta=[2, len(features_list)],
    					  gaussian_nb__sigma=[2, len(features_list)])

    # Get optimized parameters.
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=sss)
    t0 = time()
    cv.fit(features, labels)
    print 'Classifier tuning: %r' % round(time() - t0, 3)

    return cv


def set_classifier(x):
    return {
        'random_forest': RandomForestClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'logistic_reg': LogisticRegression(),
        'gaussian_nb': GaussianNB()
    }.get(x)


# Main function.

# Load the data.
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    print 'data_dict of length %d loaded successfully' % len(data_dict)

# Data exploration and removal of outliers.
data_analysis()
# Create new features.
email_fractions()

# Save data for easy output later.
my_dataset = data_dict


# Feature selection, using SelectKBest, k selected by GridSearchCV, and also using Stratify.
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=.65, stratify=labels)

select_k_best = SelectKBest()
sk_transform = select_k_best.fit_transform(features_train, labels_train)
indices = select_k_best.get_support(True)
print select_k_best.scores_

n_list = ['poi']
for index in indices:
    print 'features: %s score: %f' % (features_list[index + 1], select_k_best.scores_[index])
    n_list.append(features_list[index + 1])

# Final features list determined from SelectKBest and manual selection
n_list = ['poi', 'salary', 'total_stock_value', 'expenses', 'bonus',
          'exercised_stock_options', 'to_poi_fraction', 
          'from_poi_to_this_person', 'from_poi_fraction',
          'shared_receipt_with_poi']

# Update features_list with new values
features_list = n_list


# Test classifiers
# Tune your classifier to achieve better than .3 precision and recall using our testing script.

# Classifiers tested but not using - GaussianNB, Logistic_Regression, RandomForestClassifier, DecisionTreeClassifier

# cross_val = classifiers('decision_tree', 9, features_list)
# print 'Best parameters: ', cross_val.best_params_
# clf = cross_val.best_estimator_


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight='balanced'),
                         n_estimators=50, learning_rate=.8)



# Validate model precision, recall and F1-score
test_classifier(clf, my_dataset, features_list)


# Dump classifier, dataset, and features.

# Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
print 'Required information is saved.'

# References
print 'References Used:'
print 'http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html \n' \
		'http://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_housing.html\n' \
		'https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/ \n' \
        'http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html \n' \
        'http://scikit-learn.org/stable/modules/pipeline.html \n' \
        'http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB \n' \
