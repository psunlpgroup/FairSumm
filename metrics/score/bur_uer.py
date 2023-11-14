import math
from collections import defaultdict
# Compute BUR and UER


# This class is samplewise which means we need to initialize every sample
class BUR_UER:
    def __init__(self, CUTOFF_THRESHOLD = 1, UNFAIR_THRESHOLD = 0.8, NUM_THRESHOLD = 0):
        self.CUTOFF_THRESHOLD = CUTOFF_THRESHOLD
        self.UNFAIR_THRESHOLD = UNFAIR_THRESHOLD
        self.NUM_THRESHOLD = NUM_THRESHOLD

        self.unfair_speaker = defaultdict(int)
        self.unfair_distance = defaultdict(list)

    def cutoff_distribution(self, source, target, cutoff=None):
        if cutoff is None:
            cutoff = self.CUTOFF_THRESHOLD

        # cut off long tails
        speakers = sorted(source.keys(), key=lambda x: -source[x])
        ratio_count = 0
        source_tot = sum(source.values())
        target_tot = sum(target.values())
        new_speakers = []
        for speaker in speakers:
            ratio_count += source[speaker] / source_tot
            new_speakers.append(speaker)
            if ratio_count >= cutoff:
                break

        # assign new speakers to data
        speakers = new_speakers
        source = {x: y for x, y in source.items() if x in speakers}
        target = {x: y for x, y in target.items() if x in speakers}

        return source, target

    def compare_distribution(self, source, target):
        speakers = source.keys()
        source_tot = sum(source.values())
        target_tot = sum(target.values())
        diff = []

        for speaker in speakers:
            source_ratio = source[speaker] / source_tot
            target_ratio = target[speaker] / target_tot if target_tot else 1 / len(speakers)
            diff.append(max(0,source_ratio-target_ratio))

        return sum(diff)/len(diff)

    def compute_second_order_fairness(self, value_dict):
        values = []
        for k, v in value_dict.items():
            v = [math.fabs(x) for x in v]
            values.append(sum(v)/len(v))
        # KMeans target
        avg = lambda x: sum(x)/len(x)
        values = [math.fabs(x-avg(values)) for x in values]
        return avg(values)


    def detect_unfair(self, source, target, threshold=None, verbose=False):
        if threshold is None:
            threshold = self.UNFAIR_THRESHOLD
        speakers = sorted(source.keys(), key=lambda x: -source[x])
        source = {x:y for x,y in source.items() if x in speakers}
        target = {x:y for x,y in target.items() if x in speakers}
        source_tot = sum(source.values())
        target_tot = sum(target.values())

        # count unfair
        unfair_count = 0
        unfair_speakers = []
        unfair_ratios = []
        for speaker in speakers:
            source_ratio = source[speaker] / source_tot
            target_ratio = target[speaker] / target_tot if target_tot else 1/len(speakers)
            self.unfair_distance[speaker].append(max(0,source_ratio-target_ratio))
            if target_ratio <= source_ratio * threshold:
                unfair_count += 1
                self.unfair_speaker[speaker] += 1
            if verbose:
                print(speaker, "source ratio:", round(source_ratio,2), "target ratio:", round(target_ratio,2),
                      "Unfair" if target_ratio/source_ratio < threshold else "Fair")
            # if target_ratio / source_ratio < UNFAIR_THRESHOLD:
                unfair_speakers.append(speaker)
                unfair_ratios.append(source_ratio)
        if verbose:
            print(unfair_speakers, [round(x,2) for x in unfair_ratios])
            return unfair_count, self.unfair_speaker, unfair_ratios
        return unfair_count


    def compute_score(self, source_distribution, target_distribution):
        # Cutoff long tail distribution
        cutoff_source, cutoff_target = self.cutoff_distribution(source_distribution, target_distribution)
        diff = self.compare_distribution(cutoff_source, cutoff_target)
        unfair_count = self.detect_unfair(cutoff_source, cutoff_target)
        print("Sample's diff:", diff)
        print("Fair or not:", "Unfair" if unfair_count > self.NUM_THRESHOLD else "Fair")
        return diff, unfair_count > self.NUM_THRESHOLD