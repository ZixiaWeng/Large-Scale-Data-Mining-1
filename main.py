import pprint as pp
from sklearn.datasets import fetch_20newsgroups


def get_graphic_len(category):
    return len(fetch_20newsgroups(
        subset='train',
        categories=category,
        shuffle=True,
        random_state=42
    )['data'])

categories = [
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

res = {}
for c in categories:
    res[c] = get_graphic_len([c])

print res
# pp.pprint(get_graphic(categories    ))