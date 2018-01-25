from sklearn.datasets import fetch_20newsgroups

categories = ['comp.graphics']
graphics_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)
print graphics_train