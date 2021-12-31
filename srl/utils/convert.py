def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob_ranges(words, tags):
    """
    IOB -> Ranges
    """
    assert len(words) == len(tags)
    events = {}

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O' or tags[i + 1].split('-')[0] =='B':
            events[temp_type] = ''.join(words[begin: i + 1])

    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            temp_type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            check_if_closing_range()
    return events

def iobes_ranges(words, tags):
    new_tags = iobes_iob(tags)
    return iob_ranges(words, new_tags)

def format_result(result, text, tag):
    entities = []
    for i in result:
        begin, end = i
        entities.append({
            "start":begin,
            "stop":end + 1,
            "word":text[begin:end+1],
            "type":tag
        })
    return entities

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    tags = []

    for index_1 in range(len(path)):
        if path[index_1] == begin_tag:
            ner_index = 0
            for index_2 in range(index_1 + 1, len(path)):
                if path[index_2] == mid_tag:
                    ner_index += 1
                else:
                    break
            if ner_index != 0:
                tags.append([index_1, index_1 + ner_index])
    return tags
