from random import randint

def generateNum() -> int:
    return randint(1, 5)


def get_data():
    trees = ["aaa", "bbb", "ccc"]
    injuries = ["yes", "no"]
    answers = ["it's okay", "it's not okay"]
    data = {'event_id': 45, 'date': f'{randint(0, 2025)}-{randint(1, 12)}-{randint(1, 31)})T10:30:00Z',
            'tree_type': trees[randint(0, 2)], 'has_cracks': randint(0, 1),
            'has_hollows': randint(0, 1), 'has_fruits_or_flowers': randint(0, 1),
            'injuries': injuries[randint(0, 1)], 'photo_file_name': trees[randint(0, 2)] + ".png",
            'answer': answers[randint(0, 1)]}
    return data