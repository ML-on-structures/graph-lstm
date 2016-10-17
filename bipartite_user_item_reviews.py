class Item:
    def __init__(self, id, assignment_id = None, inherent = None, max_inherent = None):
        self.id = id
        self.assignment_id = assignment_id
        self.inherent = inherent
        self.max_inherent = max_inherent
        self.reviews = {}

class User:
    def __init__(self, name):
        self.name = name
        self.users = set()
        self.reviews = {}

class Review:
    def __init__(self, review_id = None, grade = None, extra_informative_feature = None):
        self.review_id = id
        self.grade = grade
        self.extra_informative_feature = extra_informative_feature

class Graph:
    def __init__(self):
        self.items = set()
        self.users = set()
        self.reviews = {}
        self.items_with_ground_truth = []
        self.user_dict = {}
        self.item_dict = {}

    def add_item(self, item_id, inherent = None, max_inherent = None, assignment_id = None):
        item = Item(id = item_id, inherent= inherent, max_inherent= max_inherent, assignment_id= assignment_id)
        self.item_dict[item_id] = item
        self.items = self.items | {item}

    def add_user(self, user_name):
        user = User(user_name)
        self.user_dict[user_name] = user
        self.users = self.users | {user}

    def get_user(self, user_name):
        return self.user_dict.get(user_name)

    def get_item(self, item_id):
        return self.item_dict.get(item_id)

    def has_voted(self, user_name,item_id):
        if not user_name in self.user_dict or not item_id in self.item_dict:
            return False
        if (self.get_user(user_name), self.get_item(item_id)) in self.reviews:
            return True
        else:
            return False

    def get_no_of_votes(self, item_id):
        if not item_id in self.item_dict:
            return 0
        return len(self.get_item(item_id).reviews)

    def add_review(self, user_name, item_id, review, assignment_id = None):
        """
        Adds a review to the graph.
        It inserts the review to the generic dictionary of reviews
        but also to the item.reviews and user.reviews dictionaries.
        There is redundancy of information but enhances accessibility.
        """
        # If user name or item id are not in the graph create respective objects
        if not user_name in self.user_dict:
            self.add_user(user_name)
        if not item_id in self.item_dict:
            self.add_item(item_id, assignment_id= assignment_id)
        # Get user and item objects that correspond to user name and user id
        user = self.get_user(user_name)
        item = self.get_item(item_id)
        # add review to the dictionaries
        item.reviews[user] = review
        user.reviews[item] = review
        self.reviews[(user, item)] = review