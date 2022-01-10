from gensim.models import LdaModel


def print_topics(model: LdaModel, n: int = 10, writer=print):
    writer()
    writer("Topics: ")
    for idx, topic in model.show_topics(formatted=True, num_topics=n, num_words=10):
        writer(f"- {idx}: {topic}")
        writer()
