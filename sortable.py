#!/usr/bin/env python
# Sortable Coding Challenge Entry
# February 27, 2012
# Copyright 2012 Steve Hanov (steve.hanov@gmail.com)
#
# Strategy: Use a bayesian classifier to rank the best product choices for each
# listing.  Then use a heuristic to pick the best one from this short list.
# Finally, once we have grouped the listings by product, we filter out ones
# which have prices that are too good to be true.

import math, json, sys, re, heapq, codecs

# Exchange rates from April 29, 2011, current with the sample data.
ExchangeRates = {
    "USD":0.9486,
    "CAD":1.0,
    "EUR":1.4058,
    "GBP":1.58
}

class BayesianClassifier:
    def __init__(self):
        # The Bayesian classifier is trained with labeled things that have sets
        # of features. Then, given a set of features, it can give you the most
        # likely label. The labels and features can be anything you like that
        # can be used as a key in a python dictionary.
        #
        # We pretend that we have a large 2D matrix of values, even though we
        # will represent it sparsely.  The labels are the columns, and the
        # features are the rows. If an item with a label has a certain feature,
        # then the intersection of the label and feature in the matrix will
        # have a value.  
        
        #   labels
        #  ------------
        # f|         
        # e|    .
        # a|  .  .  .
        # t|
        # u|  .    
        # r|
        # e|      .
        # s|

        # for each row, we have a mapping of columns
        self.rows = {}

        # for each column, we store the set of rows that it contains
        self.columns = {}

        # we store the sums of each row
        self.columnSums = {}
        self.totalInstances = 0

    def addInstance(self, labelId, featureIds, value = 1.):
        # given a label, and its feature set, we increment the appropriate
        # entries of the matrix.
        if labelId not in self.columns:
            self.columns[labelId] = set(featureIds)
            self.columnSums[labelId] = value * len(featureIds)
        else:
            self.columns[labelId] |= featureIds
            self.columnSums[labelId] += value * len(featureIds)

        for feature in featureIds:
            if feature not in self.rows:
                self.rows[feature] = {}

            row = self.rows[feature]

            if labelId not in row:
                row[labelId] = value
            else:
                row[labelId] += value

        self.totalInstances += 1            

    def __getLikelihood(self, labelId, featureIds):
        # return the log-likelihood of the label given the features, using
        # bayes rule with laplace smoothing.

        features = featureIds & self.columns[labelId]
        if len(features) == 0: return None

        # P(Label)
        total = math.log( self.columnSums[labelId] / self.totalInstances )

        denominator = self.columnSums[labelId] + len(self.rows)
        total += float( ( len(self.rows) - len(features) ) * math.log( 1.0 / denominator ) )
        for feature in features:
            total += math.log((self.rows[feature][labelId] + 1) / denominator)

        return total

    def classify(self, featureIds, k):
        # Tries to work out the K most likely labels given the set of features.
        # returns an array of up to K (probability,label) tuples, in order of
        # decreasing probability.
        heap = []

        for labelId in self.columns.keys():
            prob = self.__getLikelihood(labelId, featureIds)
            if prob != None:
                if len(heap) < k or prob > heap[0][0]:
                    if len(heap) == k: heapq.heappop(heap)
                    heapq.heappush( heap, (prob, labelId) )
        heap.sort(reverse=True)
        return heap

LettersRe = re.compile("[A-Za-z]")
NumbersRe = re.compile("[0-9]")
SeparatorRe = re.compile("[\- ]")
ModelRe = {}

def heuristic(product, listing):
    # The heuristic function is used to filter the results from the bayesian
    # classifier. It should return 0 if we are certain the listing is not the
    # product, and higher numbers the more certain we are that it is.

    # The specific heuristic we use is that something close to the model must
    # be in the title, and a prefix of the manufacturer must be in the
    # listing's manufacturer field or title.

    # We are especially careful when matching numbers, so that 200 does not
    # match 32003 

    title = listing["title"].lower()

    # Fuzzy-match the model number. It must not be followed by more numbers.
    # Cache the regular expression for efficiency.
    if product["model"] not in ModelRe:
        modelre = re.compile(
            '(^|[ -]|DSC)' + SeparatorRe.sub('[ -]?', product["model"]) + "($|[^0-9])", 
            re.I
        )
        ModelRe[product["model"]] = modelre
    else:
        modelre = ModelRe[product["model"]]

    manufacturer = product["manufacturer"].lower()[:4]
    l_manufacturer = listing["manufacturer"].lower()

    # If we cannot find the model in the title,
    if modelre.search(title): 
        matchedModel = product["model"]
    else:
        # if the model has more than one component, find the part with at least
        # 3 numbers and check if the title has it. Fail if none are found.
        splitmodel = SeparatorRe.split(product["model"].lower())
        if len(splitmodel) == 1: return 0

        for component in splitmodel:
            if not NumbersRe.search(component): continue
            if len(component) < 3: continue
            pos = title.find(component)
            if pos != -1 and pos + len(component) < len(title) and not title[pos+len(component)].isdigit():
                # don't match a number inside of another number.
                if component.isdigit() and pos > 0 and title[pos-1].isdigit():
                    continue
                matchedModel = component
                break
        else:
            return 0

    # If we identified the model, but the model number is just a number with no
    # letters, then to be certain we must also match the product family.
    mustHaveFamily = matchedModel.isdigit()

    if len(l_manufacturer):
        if l_manufacturer.find(manufacturer) == -1: return 0
    else:
        if title.find(manufacturer) == -1: return 0

    # more points for having the product family.
    score = 1
    familyFound = False
    if "family" in product:
        
        if title.find(product["family"].lower()) != -1:
            familyFound = True
            score += 1
        else:
            # check if the title contains all the words in the family,
            # regardless of order.
            for word in product["family"].lower().split():
                if title.find(word) == -1: break
            else:
                familyFound = True
                score += 1

    if mustHaveFamily and not familyFound:
        return 0

    # more points for being from this century
    if product["announced-date"][0] == '2':
        score += 1

    return score


SplitWordsRe = re.compile('[^A-Za-z0-9\-\.]+')
SplitLettersNumbersRe = re.compile('\d+|[a-zA-Z]+')
SplitOnUnderscoresRe = re.compile('[ _]+')

def GetFeatures(record):
    # Given a product or listing record, return a set of features.  We use a
    # bag of lower case words split on whitespace. Any words consisting of
    # letters and numbers will be split into their components and added. Words
    # in the manufacturer field will be prefixed with the string
    # "manufacturer:"

    features = []

    def add(str):
        for s in SplitOnUnderscoresRe.split(str):
            if s.find("-") != -1:
                features.append(s.replace("-", ""))
        for s in SplitWordsRe.split(str):
            features.append(s)
            features.extend( SplitLettersNumbersRe.findall(s) )

    for field in ["product_name", "model", "title", "family", "manufacturer"]:
        if field in record:
            if len(record[field]):
                add(record[field])

    if "manufacturer" in record:
        features.append( "manufacturer:" + record["manufacturer"] )

    return set([s.lower() for s in features])

def MedianDeviation(numericValues):
    # Returns the median, and the Median Absolute Deviation.
    # This is a robust measure related to deviation from the median.
    theValues = sorted(numericValues)

    if len(theValues) % 2 == 1:
        median = theValues[int(len(theValues)/2)]
    else:
        lower = theValues[int(len(theValues)/2)-1]
        upper = theValues[int(len(theValues)/2)]
        median = (float(lower + upper)) / 2      

    d = sorted([abs(i - median) for i in numericValues])
    # multiplying by 1.48 is supposed to make the median deviation more like
    # the standard deviation or something.
    if len(d) % 2 == 1:
        deviation = 1.48 * d[int(len(d)/2)]
    else:
        lower = d[int(len(d)/2)-1]
        upper = d[int(len(d)/2)]
        deviation = 1.48 * (float(lower + upper)) / 2      
    return (median, deviation)

def FilterPriceOutliers(listings, outliers):
    # The sample data of cameras included camera accessories as well. We filter
    # out items that have prices that are too good to be true. 
    # We cut out products priced too low if it is unusual for it to be that
    # low, given the other competitor's prices.
    # To do this, we calculate the median absolute deviation, and exclude
    # products priced at a 60% of the median, if this is outside of 2.5 median
    # deviations.
    prices = []
    for item in listings:
        if item["currency"] not in ExchangeRates: 
            print("Need exchange rate for " + item["currency"])
            continue
        prices.append(float(item["price"]) * ExchangeRates[item["currency"]])
    if len(prices) < 2: return listings
    (median, deviation) = MedianDeviation(prices)
 
    filtered = []
    for item in listings:
        if item["currency"] in ExchangeRates: 
            price = float(item["price"]) * ExchangeRates[item["currency"]]
            if price < median * 0.6 and price < median - deviation * 2.5:
                outliers.append(item)
                continue
        filtered.append(item)
    return filtered

def ClassifyProducts(model, products, listings, results, K):
    i = 0
    failures = 0
    unclassified = []
    for item in listings:
        if (i % 100) == 0:
            sys.stderr.write("K=%d Classified %d/%d of %d listings\r" % (K, i-failures, i, len(listings)))
            sys.stderr.flush()

        # Get a ranked list of possibilities from the classifier. Use the one
        # that scores highest with the heuristic.
        possibilities = model.classify(GetFeatures(item), K)
        bestScore = None
        bestLabel = None
        for (prob, label) in possibilities:
            score = heuristic(products[label], item)
            if bestScore == None or score > bestScore:
                bestScore = score
                bestLabel = label
                if score == 3: break

        if bestScore:
            results[bestLabel].append(item)
        else:
            failures += 1
            unclassified.append(item)
                
        i += 1
    sys.stderr.write("\n")        

    return unclassified
        

def main():
    # K is how many results to return from the bayesian classifier. They are
    # each run through the heurisitic function to score them.
    K = 182

    # The products are the known products that we are trying to match to.
    products = []

    # The listings are the uncategoristed listings from the web.
    listings = []

    # Outliers were stripped out due to suspicious prices.
    outliers = []

    # Read the files.
    sys.stderr.write("Reading listings.txt\n")
    for line in codecs.open("listings.txt", encoding="UTF-8").readlines():
        listings.append(json.loads(line))

    sys.stderr.write("Reading products.txt\n")
    for line in codecs.open("products.txt", encoding="UTF-8").readlines():
        products.append(json.loads(line))

    # train the model on the features of the products.
    sys.stderr.write("Training model\n")

    model = BayesianClassifier()
    for i in range(len(products)):
        model.addInstance(i, GetFeatures(products[i]))

    # Here is some debugging code to give insight into what the algorithm was
    # thinking. Give the program a line number in the listings file, and it
    # will display the output of the classifier and the heuristic function.
    if len(sys.argv) == 2:
        itemno = int(sys.argv[1]) - 1
        item = listings[itemno]
        print(json.dumps(item))
        print("Features: %s" % (GetFeatures(item),))
        possibilities = model.classify(GetFeatures(item), K)
        for (prob, label) in possibilities:
            score = heuristic(products[label], item )
            print("%g:%g: %s" % (prob, score, json.dumps(products[label])))
            print(GetFeatures(products[label]))
        sys.exit(0)

    sys.stderr.write("Classifying product listings\n")

    # the results array contains, for each product, an array of listings that
    # we think correspond to that product. First, build the array.
    results = []
    unclassified = []
    for i in range(len(products)):
        results.append([])

    # Finally, do the classification. For each uncategorized listing,
    unclassified = ClassifyProducts(model, products, listings, results, K)

    # Print the results.
    sys.stderr.write("\nWriting results.txt\n")
    f = open("results.txt", "wt")
    for i in range(len(products)):
        results[i] = FilterPriceOutliers(results[i], outliers)
        item = { 
            "product_name": products[i]["product_name"],
            "listings": results[i]
        }
        f.write(json.dumps(item) + "\n")

    f = open("unclassified.txt", "wt")
    for item in unclassified:
        f.write(json.dumps(item) + "\n")

    f = open("outliers.txt", "wt")
    for item in outliers:
        f.write(json.dumps(item) + "\n")

    sys.stderr.write("Filtered %d suspicious prices. See outliers.txt\n" % (len(outliers),))

main()    
