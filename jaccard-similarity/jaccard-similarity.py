def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    a = set(set_a)
    b = set(set_b)

    if not a and not b:
        return 0.0

    intersection_size = len(a & b)
    union_size = len(a | b)

    return intersection_size / union_size

    