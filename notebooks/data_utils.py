import re

def extract_contracts(dataset):
    contracts = []
    for contract in dataset['data']:
        for paragraph_index, paragraph in enumerate(contract['paragraphs']):
            contracts.append({
                "contract_id": contract['title'],
                'paragraph_index': paragraph_index,
                'text': paragraph['context'],
                'qas': paragraph['qas']
            })

    return contracts



def create_labeled_chunks(contract, tokenizer, chunk_length=128, overlap=20):
    text = contract["text"]

    if chunk_length <= 0 or not 0 <= overlap < chunk_length:
        raise ValueError("Require chunk_length > 0 and 0 <= overlap < chunk_length")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    step = chunk_length - overlap

    # Keep each answer's text and its character span in the original paragraph.
    answers = []
    for qa in contract["qas"]:
        match = re.search(r'related to "([^"]+)"', qa["question"])
        if match is None:
            raise ValueError(f"Could not find category in question: {qa['question']}")

        category = match.group(1)

        for answer in qa["answers"]:
            start = answer["answer_start"]
            end = start + len(answer["text"])

            if text[start:end] != answer["text"]:
                raise ValueError(f"Answer offset does not match: {answer['text']!r}")

            answers.append((category, start, end))

    chunks = []
    for token_start in range(0, len(input_ids), step):
        token_end = min(token_start + chunk_length, len(input_ids))
        char_start = offsets[token_start][0]
        char_end = offsets[token_end - 1][1]

        labels = {
            category
            for category, answer_start, answer_end in answers
            if answer_start < char_end and char_start < answer_end # any amount of overlap is sufficient
            # if chunk_start <= answer_start and answer_end <= chunk_end # more strict, answer must be full contained in chunk
        }

        chunks.append({
            "text": text[char_start:char_end],
            "input_ids": input_ids[token_start:token_end],
            "char_start": char_start,
            "char_end": char_end,
            "labels": labels,
        })

        if token_end == len(input_ids):
            break

    return chunks


def multihot_encode(labels, label_to_id):
    vector = [0] * len(label_to_id)
    for label in labels:
        if label in label_to_id:
            vector[label_to_id[label]] = 1
    return vector


def multihot_decode(vector, id_to_label):
    labels = []
    for i, v in enumerate(vector):
        if v == 1:
            labels.append(id_to_label[i])
    return labels