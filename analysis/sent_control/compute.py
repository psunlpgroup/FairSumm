from nltk import sent_tokenize
import json
natural_sent_data = [json.loads(x) for x in open("/scratch1/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_16.txt")]
one_sent_data = [json.loads(x) for x in open("/scratch1/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_19.txt")]
three_sent_data = [json.loads(x) for x in open("/scratch1/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_20.txt")]
five_sent_data = [json.loads(x) for x in open("/scratch1/yfz5488/fairsumm/models/GPTs/results/Claritin/predictions_21.txt")]

def parse_data(data, target_num):
    num_sents = []
    for sample in data:
        pred = sample['prediction']
        num_sent = len(sent_tokenize(pred))
        num_sents.append(num_sent)
    return sum(num_sents)/len(num_sents)
if __name__ == '__main__':
    print("Natural Sent Avg:", parse_data(natural_sent_data,3))
    print("1 sent Avg:", parse_data(one_sent_data,1))
    print("3 sent Avg:", parse_data(three_sent_data,3))
    print("5 sent Avg:", parse_data(five_sent_data,5))